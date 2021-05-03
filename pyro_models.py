import pyro
import pyro.distributions as dist
import torch
import pyro.params.param_store as param_store
import pyro.poutine as poutine


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
        names = self.paramdict.keys()

        assert "sex_prob" in names
        self.sex_prob = self.paramdict.get_param("sex_prob")

        assert "mus_male" in names
        self.mus_male = self.paramdict.get_param("mus_male")

        assert "mus_female" in names
        self.mus_female = self.paramdict.get_param("mus_female")

    def get_conditional_prob(self, *args, **kwargs):

        assert 'Mustache' in kwargs
        mustache = torch.tensor(kwargs['Mustache'])
        mustache = torch.clamp(mustache, 0)

        def deterministic_model(Male=None, Mustache=None):
            Male = pyro.sample('Male', dist.Bernoulli(self.sex_prob), obs=Male)
            mustache_prob = self.mus_male * Male + self.mus_female * (1 - Male)
            Mustache = pyro.sample('Mustache', dist.Bernoulli(mustache_prob), obs=Mustache)

        def deterministic_guide(Male=None, Mustache=None):
            pass

        enum_model = pyro.infer.config_enumerate(deterministic_model)
        elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=0)

        conditional_marginals = elbo.compute_marginals(enum_model, deterministic_guide, Mustache=mustache.float())

        sex_res = conditional_marginals['Male'].log_prob(torch.tensor(1.0)).exp().item()

        return {'Male': sex_res}

    def get_joint_prob(self, *args, **kwargs):

        assert 'Mustache' in kwargs
        mustache = torch.tensor(kwargs['Mustache'])

        male = None
        try:
            male = torch.tensor(kwargs['Male'])
        except:
            pass

        # We check if the joint probability of the original sex + mustache is over 0.7 we will keep
        # the original label for the related attributes
        if male is not None:
            sex_prob = self.sex_prob if male.item() == 1 else (1-self.sex_prob)
            mustache_prob = self.mus_male * male + self.mus_female * (1 - male) if mustache.item() == 1 else (1 - self.mus_male) * male + (1 - self.mus_female) * (1 - male)
            joint_prob = mustache_prob*sex_prob
        else:
            joint_prob = torch.tensor(0.0)

        return joint_prob


