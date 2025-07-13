import os


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

    def ask_single_record(self, feature_names: list[str]) -> dict:
        """Ask the user for a single record."""
        print("Please enter values for the following features:")
        user_input = {}
        for feature in feature_names:
            val = input(f"{feature}: ").strip()
            if not val:
                print(f"Value for {feature} cannot be empty. Try again.")
                return self.ask_single_record(feature_names)
            user_input[feature] = val
        return user_input
