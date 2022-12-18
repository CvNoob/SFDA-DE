from . import resnet
from .domain_specific_module import BatchNormDomain
from utils import utils
from . import utils as model_utils
import torch.nn as nn
import torch.nn.functional as F

backbones = [resnet]


class FC_BN_ReLU_Domain(nn.Module):
    def __init__(self, in_dim, out_dim, num_domains_bn):
        super(FC_BN_ReLU_Domain, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = BatchNormDomain(out_dim, num_domains_bn, nn.BatchNorm1d)
        self.relu = nn.ReLU(inplace=True)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AnchorNet(nn.Module):
    def __init__(self, num_classes=None,
                 feature_extractor='resnet101',
                 fx_pretrained=True, fc_hidden_dims=[], frozen=[],
                 num_domains_bn=2, dropout_ratio=(0.5,)):
        super(AnchorNet, self).__init__()
        assert num_classes is not None
        _model = utils.find_class_by_name(feature_extractor, backbones)
        self.feature_extractor = _model(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        to_select = {}
        logits, feat = self.feature_extractor(x, get_feature=True)
        to_select['feat'] = feat
        to_select['logits'] = logits
        to_select['probs'] = F.softmax(logits, dim=1)
        return to_select

    def classify(self, feat):
        # remember to freeze fc.
        to_select = {'feat': feat}
        logits = self.feature_extractor.FC(feat)
        to_select['logits'] = logits
        to_select['probs'] = F.softmax(logits, dim=1)
        return to_select

    def get_classifier_weights(self):
        return self.feature_extractor.FC.weight.data.detach().clone()

    def set_bn_domain(self, domain=0):
        pass


def anchornet(num_classes, feature_extractor,
              fx_pretrained=True, frozen=[], dropout_ratio=0.5,
              feature_extractor_state_dict=None,
              fc_hidden_dims=[], num_domains_bn=1, **kwargs):
    model = AnchorNet(feature_extractor=feature_extractor,
                      num_classes=num_classes, frozen=frozen,
                      fx_pretrained=fx_pretrained,
                      dropout_ratio=dropout_ratio,
                      fc_hidden_dims=fc_hidden_dims,
                      num_domains_bn=num_domains_bn, **kwargs)

    if feature_extractor_state_dict is not None:
        model_utils.init_weights(model.feature_extractor,
                                 feature_extractor_state_dict['state_dict'],
                                 num_domains_bn, False)

    return model
