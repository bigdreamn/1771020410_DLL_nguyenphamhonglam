from flask import Flask, render_template_string
import pandas as pd
import os

app = Flask(__name__)

template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Tables Viewer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body class="bg-light">
<div class="container py-4">
    <h1 class="mb-4">All Data Tables</h1>
    {% for name, table in tables.items() %}
        <h4 class="mt-4">{{ name }}</h4>
        <div style="max-height:400px;overflow:auto;">{{ table|safe }}</div>
    {% endfor %}
</div>
</body>
</html>
'''

@app.route('/')
def show_tables():
    tables = {}
    # Directories to search for dataset output files produced by notebooks
    dirs = [
        ("Reports", "outputs/reports"),
        ("Tables", "outputs/tables"),
        ("Figures", "outputs/figures"),
        ("Processed Data", "data/processed"),
        ("Raw Data", "data/raw")
    ]

    for group, dirpath in dirs:
        if not os.path.exists(dirpath):
            continue

        for root, _, files in os.walk(dirpath):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                key = f"{group}: {os.path.relpath(fpath, start=os.getcwd())}"

                if fname.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(fpath)
                    except Exception as e:
                        tables[key] = f'<i>Error reading CSV: {e}</i>'
                        continue
                elif fname.lower().endswith('.parquet'):
                    try:
                        df = pd.read_parquet(fpath)
                    except Exception as e:
                        tables[key] = f'<i>Error reading Parquet: {e}</i>'
                        continue
                else:
                    continue

                nrows, ncols = df.shape
                # Show all rows for small datasets, otherwise show an informative truncation
                if nrows <= 2000:
                    html_table = df.to_html(classes='table table-striped', index=False)
                    info = f"<p><strong>{nrows}</strong> rows, <strong>{ncols}</strong> columns</p>"
                else:
                    html_table = df.head(2000).to_html(classes='table table-striped', index=False)
                    info = f"<p><strong>{nrows}</strong> rows, <strong>{ncols}</strong> columns (showing first 2000 rows)</p>"

                tables[key] = info + html_table

    if not tables:
        tables['No tables found'] = '<p>No CSV or Parquet tables found in outputs/reports, outputs/tables, outputs/figures, data/processed, data/raw.</p>'

    return render_template_string(template, tables=tables)

if __name__ == '__main__':
    app.run(debug=True)
