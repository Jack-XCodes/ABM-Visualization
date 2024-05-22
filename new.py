import numpy as np
import sciris as sc  # Utilities
import starsim as ss
from starsim.diseases.sir import SIR
import pylab as pl  # Plotting

__all__ = ['Meningitis']

class Meningitis(SIR):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """
        Initialize with parameters
        """
        super().__init__(pars=pars, **kwargs)
        self.default_pars = {
            'dur_exp_inf': 2,  # (days)
            'dur_exp_rec': 2,  # (days)
            'dur_inf': 14,  # (days)
            'dur_rec': 7,  # (days)
            'p_death': 0.05,  # (prob of death)
            'p_symptoms': 0.4,  # probability of showing symptoms
            'init_prev': 0.005,  # Init cond
            'beta': 0.08,  # Init cond
            'rel_beta_inf': 0.5,  # Reduction in transmission for I versus E
            'waning': 1 / (365 * 3),
            'imm_boost': 0.001,
        }
        self.update_pars(pars, **kwargs)

        self.par_dists = ss.omergeleft(par_dists,
            dur_exp_inf=ss.normal,
            dur_exp_rec=ss.normal,
            dur_inf=ss.normal,
            dur_rec=ss.normal,
            init_prev=ss.bernoulli,
            p_death=ss.bernoulli,
            p_symptoms=ss.bernoulli,
            imm_boost=ss.delta
        )

        # SIR are added automatically, here we add E
        self.add_states(
            ss.State('exposed', bool, False),
            ss.State('ti_exposed', float, np.nan),
            ss.State('ti_recovered', float, np.nan),
            ss.State('ti_susceptible', float, np.nan),
            ss.State('immunity', float, 0.0),
        )
        return

    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'rel_sus', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_recoveries', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'recovered', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'exposed', sim.npts, dtype=float)
        return

    def update_results(self, sim):
        """ Store the population immunity (susceptibility) """
        super().update_results(sim)
        self.results['rel_sus'][sim.ti] = self.rel_sus.mean()
        self.results['new_recoveries'][sim.ti] = np.count_nonzero(self.ti_recovered == sim.ti)
        self.results['recovered'][sim.ti] = np.count_nonzero(self.immunity > 1)
        self.results['exposed'][sim.ti] = self.exposed.sum()
        return 

    @property
    def infectious(self):
        return self.infected | self.exposed

    def update_pre(self, sim):
        # Progress exposed -> recovered
        exposed_recovered = ss.true(self.exposed & (self.ti_recovered <= sim.ti))
        self.exposed[exposed_recovered] = False
        self.recovered[exposed_recovered] = True
        
        # Progress exposed -> infected
        infected = ss.true(self.exposed & (self.ti_infected <= sim.ti))
        self.exposed[infected] = False
        self.infected[infected] = True

        # Progress infected -> recovered
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Progress recovered -> susceptible
        susceptible = ss.true(self.recovered & (self.ti_susceptible <= sim.ti))
        self.recovered[susceptible] = False
        self.susceptible[susceptible] = True
        self.update_immunity(sim)

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.ti)
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def update_immunity(self, sim):
        uids = ss.true(self.immunity > 0)
        self.immunity[uids] = (self.immunity[uids]) * (1 - self.pars.waning * sim.dt)
        self.rel_sus[uids] = np.maximum(0, 1 - self.immunity[uids])
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        # Do not call set_prognosis on parent
        # super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = sim.ti

        p = self.pars
        self.immunity[uids] += p.imm_boost.rvs(uids)

        # Determine who will develop symptoms
        has_symptoms = p.p_symptoms.rvs(uids)
        symptomatic_uids = uids[has_symptoms]
        carrier_uids = uids[~has_symptoms]

        # Determine when exposed carriers recover
        self.ti_recovered[carrier_uids] = sim.ti + p.dur_exp_rec.rvs(carrier_uids) / sim.dt
        self.ti_susceptible[carrier_uids] = self.ti_recovered[carrier_uids] + p.dur_rec.rvs(carrier_uids) / sim.dt

        # Determine when exposed become infected for those who develop symptoms
        self.ti_infected[symptomatic_uids] = sim.ti + p.dur_exp_inf.rvs(symptomatic_uids) / sim.dt

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(symptomatic_uids)
        dur_rec = p.dur_rec.rvs(symptomatic_uids)
        
        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(symptomatic_uids)
        dead_uids = symptomatic_uids[will_die]
        rec_uids = symptomatic_uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die] / sim.dt
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die] / sim.dt
        self.ti_susceptible[rec_uids] = self.ti_recovered[rec_uids] + dur_rec[~will_die] / sim.dt

        return

    def update_death(self, sim, uids):
        # Reset infected/recovered flags for dead agents
        for state in ['susceptible', 'exposed', 'infected', 'recovered']:
            self.statesdict[state][uids] = False
        return

    def make_new_cases(self, sim):
        """
        Add new cases of module, through transmission, incidence, etc.
        
        Common-random-number-safe transmission code works by mapping edges onto
        slots.
        """
        new_cases = []
        sources = []
        people = sim.people
        beta = self.pars.beta[0][0]

        net = sim.networks[0]
        contacts = net.contacts
        rel_trans = (self.infectious & people.alive) * self.rel_trans
        rel_trans[self.infected] *= self.pars.rel_beta_inf  # Modify transmissibility of people with symptoms
        rel_sus = (self.susceptible & people.alive) * self.rel_sus
        p1p2b0 = [contacts.p1, contacts.p2]
        p2p1b1 = [contacts.p2, contacts.p1]
        for src, trg in [p1p2b0, p2p1b1]:

            # Calculate probability of a->b transmission.
            beta_per_dt = net.beta_per_dt(disease_beta=beta, dt=people.dt)  # TODO: should this be sim.dt?
            p_transmit = rel_trans[src] * rel_sus[trg] * beta_per_dt

            # Generate a new random number based on the two other random numbers -- 3x faster than `rvs = np.remainder(rvs_s + rvs_t, 1)`
            rvs_s = self.rng_source.rvs(src)
            rvs_t = self.rng_target.rvs(trg)
            rvs = rvs_s + rvs_t
            inds = np.where(rvs > 1.0)[0]
            rvs[inds] -= 1
            
            new_cases_bool = rvs < p_transmit
            new_cases.append(trg[new_cases_bool])
            sources.append(src[new_cases_bool])
                
                # Tidy up
        if len(new_cases) and len(sources):
            new_cases = np.concatenate(new_cases)
            sources = np.concatenate(sources)
        else:
            new_cases = np.empty(0, dtype=int)
            sources = np.empty(0, dtype=int)
            
        if len(new_cases):
            self._set_cases(sim, new_cases, sources)
            
        return new_cases, sources
        
    def plot(self):
        """ Default plot for SEIRS model """
        fig = pl.figure()
        for rkey in ['susceptible', 'exposed', 'infected', 'recovered']:
            pl.plot(self.results['n_'+rkey], label=rkey.title())
        pl.legend()
        pl.close()
        return fig