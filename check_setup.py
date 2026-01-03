import os
import glob

print("🔍 CHECKING YOUR SETUP\n")
print("="*60)

# Check app.py
if os.path.exists('app.py'):
    print("✅ app.py found")
else:
    print("❌ app.py NOT found")

# Check models folder
models_dir = 'models'
if os.path.exists(models_dir):
    keras_files = glob.glob(f'{models_dir}/*_model.keras')
    pkl_files = glob.glob(f'{models_dir}/*_scaler.pkl')
    
    print(f"\n📁 MODELS FOLDER:")
    print(f"   ✅ Found {len(keras_files)} model files (.keras)")
    print(f"   ✅ Found {len(pkl_files)} scaler files (.pkl)")
    
    if len(keras_files) > 0:
        print(f"\n   Sample models:")
        for f in sorted(keras_files)[:5]:
            stock = os.path.basename(f).replace('_model.keras', '')
            scaler = f.replace('_model.keras', '_scaler.pkl')
            if os.path.exists(scaler):
                print(f"      ✅ {stock} (model + scaler)")
            else:
                print(f"      ⚠️  {stock} (model only, scaler missing!)")
else:
    print(f"\n❌ MODELS FOLDER NOT FOUND")
    print(f"   Create: mkdir models")

# Check data folder
data_dir = 'data'
if os.path.exists(data_dir):
    csv_files = glob.glob(f'{data_dir}/*.csv')
    
    print(f"\n📁 DATA FOLDER:")
    print(f"   ✅ Found {len(csv_files)} CSV files")
    
    if len(csv_files) > 0:
        print(f"\n   Sample data files:")
        for f in sorted(csv_files)[:5]:
            print(f"      - {os.path.basename(f)}")
else:
    print(f"\n❌ DATA FOLDER NOT FOUND")
    print(f"   Create: mkdir data")

# Check if models match data
if os.path.exists(models_dir) and os.path.exists(data_dir):
    print(f"\n🔗 CHECKING MATCHES:")
    
    stock_models = set([os.path.basename(f).replace('_model.keras', '') 
                       for f in glob.glob(f'{models_dir}/*_model.keras')])
    stock_data = set([os.path.basename(f).replace('.csv', '') 
                     for f in glob.glob(f'{data_dir}/*.csv')])
    
    matched = stock_models & stock_data
    models_only = stock_models - stock_data
    data_only = stock_data - stock_models
    
    print(f"   ✅ Matched (model + data): {len(matched)} stocks")
    
    if models_only:
        print(f"   ⚠️  Models without data: {len(models_only)}")
        print(f"      {', '.join(sorted(list(models_only))[:5])}")
    
    if data_only:
        print(f"   ⚠️  Data without models: {len(data_only)}")
        print(f"      {', '.join(sorted(list(data_only))[:5])}")

print("\n" + "="*60)
print("✅ Setup check complete!")
print("\nTo run your app: streamlit run app.py")