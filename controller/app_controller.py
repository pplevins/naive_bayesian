from loader import DataLoader
from core import CategoricalNaiveBayes
from service import RecordClassifier


class AppController:
    """A manager class to control the flow of the application."""

    def __init__(self, ui):
        """Constructor."""
        self.ui = ui
        self.data_loader = DataLoader()
        self.classifier = CategoricalNaiveBayes()

    def run(self):
        """Run the application."""
        path = self.ui.ask_file_path()
        dataset = self.data_loader.load_and_encode(path)

        train_set, test_set = dataset.split()
        self.classifier.fit(train_set.X, train_set.y)
        self.ui.show_message("Model trained successfully.")

        classifier_service = RecordClassifier(self.classifier)

        mode = self.ui.ask_mode()
        if mode == "batch":
            accuracy = classifier_service.evaluate_accuracy(test_set)
            self.ui.show_message(f"Model Accuracy: {accuracy:.2%}")
        elif mode == "single":
            self.handle_single_record(classifier_service)

    def handle_single_record(self, classifier_service):
        feature_names = self.data_loader.get_feature_names()
        try:
            raw_input = self.ui.ask_single_record(feature_names)
            encoded = self.data_loader.encoder_util.transform_single_record(raw_input, feature_names)
            prediction_encoded = classifier_service.classify_single(encoded)
            label_encoder = self.data_loader.encoder_util.get_encoder("class")
            prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
            self.ui.show_message(f"Predicted class: {prediction_decoded}")
        except ValueError as e:
            self.ui.show_message(f"Error: {e}")
            self.handle_single_record(classifier_service)  # Retry
