import os

from loader import DataLoader


class CLIInterface:
    """A class representing a CLI interface for the UI."""

    def show_message(self, msg):
        """Show a message to the user."""
        print(msg)

    def ask_file_path(self):
        """Ask the user for a file path."""
        while True:
            path = input("Enter CSV file path: ").strip()
            if os.path.exists(path) and path.endswith(".csv"):
                return path
            print("Invalid path or file not found. Please try again.")

    def ask_mode(self):
        """Ask the user for a mode."""
        while True:
            mode = input("Choose mode - 'batch' or 'single': ").strip().lower()
            if mode in ("batch", "single"):
                return mode
            print("Invalid input. Please enter 'batch' or 'single'.")

    def ask_single_record(self, data_loader: DataLoader) -> dict:
        """Ask the user for a single record."""
        print("Please enter values for the following features:")
        user_input = {}
        feature_names = data_loader.get_feature_names()
        for feature in feature_names:
            feature_values = data_loader.get_feature_unique_values(feature)
            val = input(f"{feature} - {feature_values}: ").strip()
            if not val:
                print(f"Value for {feature} cannot be empty. Try again.")
                return self.ask_single_record(data_loader)
            user_input[feature] = val
        return user_input
