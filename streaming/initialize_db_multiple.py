from django.db import connections

DATABASES = {
    'main': 'default',  # Base principale
    'people': 'people',  # Base secondaire 1
    'vehicles': 'vehicles',  # Base secondaire 2
    'garbage': 'garbage'   # Base secondaire 3
}

TABLES = [
    'api_detection',
    'api_boundingbox',
    'api_center',
    'api_inferenceresult',
    'api_imagesize',
]


def truncate_tables(db_name):
    """
    Vide toutes les tables d'une base de données spécifique.
    """
    with connections[db_name].cursor() as cursor:
        cursor.execute("SET CONSTRAINTS ALL DEFERRED;")  # Désactiver temporairement les contraintes
        for table in TABLES:
            cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
        print(f"Toutes les tables ont été vidées dans la base {db_name}.")


def clear_all_databases():
    """
    Vide les tables dans toutes les bases (principale et secondaires).
    """
    for db_name in DATABASES.values():
        truncate_tables(db_name)
    print("Toutes les bases de données ont été vidées avec succès.")
