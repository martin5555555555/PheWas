from codes.models.Transformers.Transformer_V1 import TransformerGeneModel_V1
from codes.models.Transformers.Transformer_V2 import TransformerGeneModel_V2

DIC_MODEL_VERSIONS = {
    'transformer_V1' : TransformerGeneModel_V1,
    'transformer_V2' : TransformerGeneModel_V2
}