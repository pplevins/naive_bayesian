from controller import AppController
from ui import CLIInterface


def main():
    """Main entry point."""
    ui = CLIInterface()
    app = AppController(ui)
    app.run()


if __name__ == "__main__":
    main()
