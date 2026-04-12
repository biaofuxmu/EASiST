from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2Config

try:
    from .modeling_wav2vec_s import Wav2VecSModel
    from .configuration_wav2vec_s import Wav2VecSConfig
except:
    from modeling_wav2vec_s import Wav2VecSModel
    from configuration_wav2vec_s import Wav2VecSConfig


SpeechModels = {
    "wav2vec_s": Wav2VecSModel,
    "wav2vec2": Wav2Vec2Model,
}

SpeechConfigs = {
    "wav2vec_s": Wav2VecSConfig,
    "wav2vec2": Wav2Vec2Config
}

SpeechFeatureExtractors = {
    "wav2vec_s": Wav2Vec2FeatureExtractor,
    "wav2vec2": Wav2Vec2FeatureExtractor
}