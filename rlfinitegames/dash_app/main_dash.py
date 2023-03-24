import i18n
from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
from rlfinitegames.dash_app.source.components.layout import create_layout


LOCALE = "de"


def main() -> None:
    # set the locale and load the translations

    i18n.set("locale",LOCALE)
    i18n.load_path.append("rlfinitegames/dash_app/locale")

    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = i18n.t("general.app_title")
    app.layout = create_layout(app)
    app.run()

# Run the app
if __name__ == '__main__':
    main()
