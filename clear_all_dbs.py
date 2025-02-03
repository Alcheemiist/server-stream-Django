import os
import django
from django.core.management.base import BaseCommand
from django.db import connections

# Initialiser Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')  # Remplace par le chemin vers ton fichier settings.py
django.setup()

# Définir les bases de données et les tables à vider
DATABASES = {
    'main': 'default',  # Base principale
    'people': 'people',  # Base secondaire 1
    'vehicles': 'vehicles',  # Base secondaire 2
    'garbage': 'garbage'   # Base secondaire 3
}

TABLES = [
    'streaming_detection',
    'streaming_boundingbox',
    'streaming_center',
    'streaming_inferenceresult',
    'streaming_imagesize',
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

class Command(BaseCommand):
    help = "Vide toutes les tables dans toutes les bases de données."

    def handle(self, *args, **kwargs):
        clear_all_databases()
        self.stdout.write(self.style.SUCCESS("Toutes les bases de données ont été vidées."))

if __name__ == "__main__":
    clear_all_databases()
