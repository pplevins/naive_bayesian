from controller import api_client


class AppController:
    """A manager class to control the flow of the application."""

    def __init__(self, ui):
        """Constructor."""
        self.ui = ui

    def run(self):
        """Run the application."""
        self.ui.show_message("Welcome to the Naive-Bayes Classifier.")

        self.train()
        while True:
            mode = self.ui.ask_mode()
            if mode == "batch":
                self.handle_batch()
            elif mode == "single":
                self.handle_single_record()
            elif mode == "train":
                self.train()
            elif mode == "exit":
                self.ui.show_message("Goodbye.")
                break

    def train(self):
        """Train the classifier."""
        try:
            path = self.ui.ask_file_path()
            self.ui.show_message("Training started.")
            result = api_client.train_model(path)
            self.ui.show_message(f"{result['message']}\nAccuracy: {result['accuracy']:.2%}")
        except Exception as e:
            self.ui.show_message(f"Error: {e}")
            self.train()

    # def test(self, classifier_service, test_set):
    #     """Test the classifier."""
    #     accuracy = classifier_service.evaluate_accuracy(test_set)
    #     self.ui.show_message(f"Model Accuracy: {accuracy:.2%}")

    def handle_batch(self):
        """Handle a batch of data."""
        try:
            path = self.ui.ask_file_path()
            result = api_client.classify_batch(path)
            self.ui.show_message(f"Predictions:\n{result['predictions']}\n\nAccuracy: {result['accuracy']:.2%}")
        except Exception as e:
            self.ui.show_message(f"Error: {e}")
            return

    def handle_single_record(self):
        """Handle a single record classification."""
        try:
            feature_dict = api_client.get_record_values()
            raw_input = self.ui.ask_single_record(feature_dict)
            prediction = api_client.classify_single_record(raw_input)
            self.ui.show_message(f"Predicted class: {prediction}")
        except Exception as e:
            self.ui.show_message(f"Error: {e}")
            self.handle_single_record()  # Retry
