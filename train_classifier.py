from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.FeatureEngineeringModels import FeatureEngineeringModels
from classifier.training.ClassifierTraining import ClassifierTraining

if __name__ == '__main__':
    # Create Objects
    featureEngineeringModels = FeatureEngineeringModels()

    for model_type in ["video_snippet", "video_transcript", "video_comments"]:
        print("Training word vectors of ", model_type)
        featureEngineeringModels.finetune_model(model_type=model_type)

    # Create ClassifierTraining Object
    classifierTrainingObject = ClassifierTraining()

    classifierTrainingObject.train_model()
