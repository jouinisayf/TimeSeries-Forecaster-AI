import logging
from src.preprocess import load_time_series
from src.models import forecast_combined
from src.visualize import plot_forecast

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("TimeSeriesForecaster")


def main():
    """
    Pipeline complet d'analyse et de prévision d'une série temporelle.

    Cette fonction exécute les étapes suivantes :

    1. Chargement et nettoyage de la série temporelle depuis le fichier CSV;
    2. Entraînement du modèle ARIMA pour capturer la structure temporelle;
    3. Entraînement du modèle GARCH pour modéliser la volatilité;
    4. Génération d'une prévision sur un horizon défini;
    5. Affichage des résumés statistiques ARIMA et GARCH;
    6. Visualisation du graphique contenant la série originale et la prévision.

    Returns
    -------
    None
        La fonction n’a pas de valeur de retour. Elle affiche les résultats
        dans la console ainsi qu’un graphique Matplotlib.

    Notes
    -----
    Niveaux de logs utilisés :
        - INFO : étapes principales du pipeline (chargement, entraînement, affichage)
        - DEBUG : informations détaillées, notamment les valeurs prédites

    Exemple
    -------
    >>> main()
    Exécute le pipeline complet de prévision sur les prix du GLD.
    """
    logger.info("Starting forecasting pipeline")

    series = load_time_series("data/gld_price_data.csv", column="GLD")
    logger.info("Data successfully loaded")

    forecast, arima_res, garch_res = forecast_combined(series, (2, 0, 3), (1, 1), 20)

    logger.info("Models trained successfully")
    logger.debug(f"Forecast array: {forecast}")

    print("ARIMA Summary:")
    print(arima_res.summary())
    print("\nGARCH Summary:")
    print(garch_res.summary())

    logger.info("Plotting forecast")
    plot_forecast(series, forecast, 20)

    logger.info("Pipeline executed successfully")


if __name__ == "__main__":
    main()
