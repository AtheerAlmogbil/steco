from transformers import ImageGPTFeatureExtractor, ImageGPTForCausalImageModeling

from transformers import PreTrainedTokenizer, PreTrainedModel
from config import Settings


def get_model(settings: Settings) -> PreTrainedModel:
    if settings.task == 'image':
        if settings.model_name == 'openai/imagegpt-small':
            model = ImageGPTForCausalImageModeling.from_pretrained(settings.model_name).to(settings.device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    model.eval()
    return model





def get_feature_extractor(settings: Settings) -> ImageGPTFeatureExtractor:
    assert settings.task == 'image'
    if settings.model_name == 'openai/imagegpt-small':
        feature_extractor = ImageGPTFeatureExtractor.from_pretrained(settings.model_name)  # local_files_only=True
    else:
        raise NotImplementedError
    return feature_extractor
