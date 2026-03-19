from flask import Flask, render_template_string, send_from_directory
import pandas as pd
import plotly.express as px
import os

app = Flask(__name__)

# Simple HTML template with Bootstrap for a modern look
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HR Analytics Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-light">
<div class="container py-4">
    <h1 class="mb-4">HR Analytics Data Mining Dashboard</h1>
    <h3>1. EDA Summary</h3>
    <div>{{ eda_table|safe }}</div>
    <hr>
    <h3>2. Clustering Results</h3>
    <div>{{ cluster_table|safe }}</div>
    {% if cluster_fig %}
    <div id="cluster-plot"></div>
    <script>Plotly.newPlot('cluster-plot', {{ cluster_fig|safe }});</script>
    {% endif %}
    <hr>
    <h3>3. Association Rules (Top 10)</h3>
    <div>{{ assoc_table|safe }}</div>
    <hr>
    <h3>4. Supervised Model Metrics</h3>
    <div>{{ sup_table|safe }}</div>
    <hr>
    <h3>5. Semi-supervised Comparison</h3>
    <div>{{ semi_table|safe }}</div>
</div>
</body>
</html>
'''

@app.route('/')
def dashboard():
    # EDA
    eda_path = 'outputs/reports/eda_summary.csv'
    eda_table = pd.read_csv(eda_path).to_html(classes='table table-striped', index=False) if os.path.exists(eda_path) else '<i>No data</i>'

    # Clustering
    cluster_path = 'outputs/reports/clustering_results.csv'
    cluster_table = pd.read_csv(cluster_path).head(20).to_html(classes='table table-striped', index=False) if os.path.exists(cluster_path) else '<i>No data</i>'
    cluster_fig = None
    if os.path.exists(cluster_path):
        dfc = pd.read_csv(cluster_path)
        if dfc.shape[1] >= 3:
            fig = px.scatter(dfc, x=dfc.columns[1], y=dfc.columns[2], color='cluster', title='Cluster Scatter')
            cluster_fig = fig.to_json()

    # Association rules
    assoc_path = 'outputs/reports/association_rules_resigned.csv'
    assoc_table = pd.read_csv(assoc_path).head(10).to_html(classes='table table-striped', index=False) if os.path.exists(assoc_path) else '<i>No data</i>'

    # Supervised
    sup_path = 'outputs/reports/supervised_metrics.csv'
    sup_table = pd.read_csv(sup_path).to_html(classes='table table-striped', index=False) if os.path.exists(sup_path) else '<i>No data</i>'

    # Semi-supervised
    semi_path = 'outputs/reports/semi_supervised_comparison.csv'
    semi_table = pd.read_csv(semi_path).to_html(classes='table table-striped', index=False) if os.path.exists(semi_path) else '<i>No data</i>'

    return render_template_string(template,
        eda_table=eda_table,
        cluster_table=cluster_table,
        cluster_fig=cluster_fig,
        assoc_table=assoc_table,
        sup_table=sup_table,
        semi_table=semi_table
    )

if __name__ == '__main__':
    app.run(debug=True)
