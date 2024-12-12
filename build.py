import os
import shutil
import PyInstaller.__main__

def build_exe():
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')

    # Run PyInstaller with spec file only
    PyInstaller.__main__.run([
        'expense_tracker.spec'
    ])

    # Copy database to dist folder if it exists
    if os.path.exists('expense_tracker.db'):
        data_dir = os.path.join('dist', 'Expense_Tracker', 'data')
        os.makedirs(data_dir, exist_ok=True)
        shutil.copy2('expense_tracker.db', os.path.join(data_dir, 'expense_tracker.db'))

    print("Build completed! Check the 'dist' folder for your executable.")

if __name__ == "__main__":
    build_exe() 