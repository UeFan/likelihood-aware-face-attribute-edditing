import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import config_enumerate, infer_discrete


class pyro_model:

    '''
    Parent class for all pyro model used in our project
    '''

    def __init__(self, paramdict):
        '''
        paramdict is the dictionary of the probability parameters we learned. eg. 'sex_prob', 'mus_male'
        '''
        self.paramdict = paramdict

    def get_conditional_prob(self, *args, **kwargs):
        '''
        We give the traget modified label, and will receive all the related label and its corresponding postier probability
        In the form of dicitionary
        '''
        pass

    def get_joint_prob(self, *args, **kwargs):
        '''
        Return the joint probability of the target modified labels and the current label of our images.
        '''
        pass


class sex_mus_model(pyro_model):

    def __init__(self, paramdict):
        super().__init__(paramdict)

        self.paramdict = paramdict


    def determined_model(self):
        laten_sex = self.paramdict.get_param('latent_sex')

        laten_mus_1 = self.paramdict.get_param("latent_female_mustache")
        laten_mus_2 = self.paramdict.get_param("latent_male_mustache")
        laten_mus = [laten_mus_1, laten_mus_2]

        laten_beard_1 = self.paramdict.get_param("latent_no_mustache_beard")
        laten_beard_2 = self.paramdict.get_param("latent_mustache_beard")
        laten_beard = [laten_beard_1, laten_beard_2]

        laten_young = self.paramdict.get_param('latent_young')

        laten_bald_1 = self.paramdict.get_param("latent_female_old_bald")
        laten_bald_2 = self.paramdict.get_param("latent_female_young_bald")
        laten_bald_3 = self.paramdict.get_param("latent_male_old_bald")
        laten_bald_4 = self.paramdict.get_param("latent_male_young_bald")
        laten_bald = [[laten_bald_1, laten_bald_2], [laten_bald_3, laten_bald_4]]

        with pyro.plate("a_plate", size=1, dim=-2):
            sex = pyro.sample('sex', dist.Bernoulli(laten_sex))
            young = pyro.sample('young', dist.Bernoulli(laten_young))
            #         with pyro.plate("b_plate", size=1, dim=-1):

            mus = pyro.sample("mustache", dist.Bernoulli(laten_mus[sex.long()]))
            beard = pyro.sample("beard", dist.Bernoulli(laten_beard[mus.long()]))

            bald = pyro.sample("bald", dist.Bernoulli(laten_bald[sex.long()][young.long()]))

    def make_log_joint(self):
        def _log_joint(data, *args, **kwargs):
            conditioned_model = poutine.condition(self.determined_model, data=data)

            trace = poutine.trace(conditioned_model).get_trace(*args, **kwargs)

            return trace.log_prob_sum()

        return _log_joint


