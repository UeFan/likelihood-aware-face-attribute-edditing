import pyro
import pyro.distributions as dist
import torch
import pyro.params.param_store as param_store
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

        laten_makeup_1 = self.paramdict.get_param("latent_female_makeup")
        laten_makeup_2 = self.paramdict.get_param("latent_male_makeup")
        laten_makeup = [laten_makeup_1, laten_makeup_2]

        laten_young = self.paramdict.get_param('latent_young')

        laten_ear_1 = self.paramdict.get_param("latent_female_no_makeup_ear")
        laten_ear_2 = self.paramdict.get_param("latent_female_makeup_ear")
        laten_ear_3 = self.paramdict.get_param("latent_male_no_makeup_ear")
        laten_ear_4 = self.paramdict.get_param("latent_male_makeup_ear")
        laten_ear = [[laten_ear_1, laten_ear_2], [laten_ear_3, laten_ear_4]]

        laten_bag_1 = self.paramdict.get_param("latent_old_no_makeup_ear")
        laten_bag_2 = self.paramdict.get_param("latent_old_makeup_ear")
        laten_bag_3 = self.paramdict.get_param("latent_young_no_makeup_ear")
        laten_bag_4 = self.paramdict.get_param("latent_young_makeup_ear")
        laten_bag = [[laten_bag_1, laten_bag_2], [laten_bag_3, laten_bag_4]]
        with pyro.plate("a_plate", size=1, dim=-2):
            sex = pyro.sample('sex', dist.Bernoulli(laten_sex))
            young = pyro.sample('young', dist.Bernoulli(laten_young))
            with pyro.plate("b_plate", size=1, dim=-1):
                pyro.sample("mustache", dist.Bernoulli(laten_mus[sex.long()]))
                makeup = pyro.sample("makeup", dist.Bernoulli(laten_makeup[sex.long()]))

                ear = pyro.sample("ear", dist.Bernoulli(laten_ear[sex.long()][makeup.long()]))
                bag = pyro.sample("bag", dist.Bernoulli(laten_bag[young.long()][makeup.long()]))

    def make_log_joint(self):
        def _log_joint(data, *args, **kwargs):
            conditioned_model = poutine.condition(self.determined_model, data=data)

            trace = poutine.trace(conditioned_model).get_trace(*args, **kwargs)

            return trace.log_prob_sum()

        return _log_joint


