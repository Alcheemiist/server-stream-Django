import json
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

# 📌 Charger Django
django.setup()
from django.db import connections
from streaming.models import InferenceResult, DeviceZone, ImageSize
from streaming.save_data_service import save_data_to_main_db
from streaming.database_transfer import filter_and_transfer_data









# 📌 Charger le fichier JSON
file_path = "test_data.json"

if not os.path.exists(file_path):
    print(f"❌ Fichier {file_path} introuvable. Vérifie son emplacement.")
    exit()

with open(file_path, "r") as file:
    test_data = json.load(file)

# 📌 Vérifier si le DeviceZone existe, sinon le créer
device_name = test_data[0]["device"]
device_zone = DeviceZone.objects.filter(device_id=device_name).first()

if not device_zone:
    print(f"⚠️ DeviceZone introuvable pour {device_name}, création automatique...")
    device_zone = DeviceZone.objects.create(
        device_id=device_name,
        name=device_name,
        width=640,
        height=480
    )

# 📌 Enregistrer les données dans la base principale
print("📌 Test : Enregistrement de l'inférence dans `default`...")
save_data_to_main_db(test_data)

# 📌 Vérifier si l'inférence a bien été enregistrée
inference = InferenceResult.objects.filter(device=device_name).first()
if inference:
    print(f"✅ Inférence enregistrée : {inference}")
else:
    print("❌ Erreur : Inférence non enregistrée.")

# 📌 Tester le transfert vers les bases secondaires
print("📌 Test : Transfert des données vers les bases secondaires...")
filter_and_transfer_data(inference)

# 📌 Vérifier les insertions dans les bases secondaires
for db_name in ["people", "vehicles", "garbage"]:
    with connections[db_name].cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM streaming_inferenceresult")
        count = cursor.fetchone()[0]
        print(f"📊 Base `{db_name}` : {count} inférences enregistrées.")

print("🚀 Test terminé avec succès.")
