### Setup

The required libraries can be found in **misc/requirements.txt** and installed via pip as:

`pip install -r misc/requirements.txt`

Running with GPU is highly recommended.

### Dataset Generation

You can easily generate a dataset with topics of any size in **generate_dataset.ipynb**. Just use topic constants at the beginning (see comments for more). Then, run all cells and use the last cell to save the dataset with your name.

Moreover, you can add your own topics with custom noise removal (the notebook is modular). Just repeat the code in any other cell relating to a topic dataset (with required modifications).

### Evaluation Pipeline

In **clustering.ipynb**, you can use existing parts of the pipeline (text embeddings, dimensionality reduction techniques, and clustering algorithms) with your own parameters or easily add new ones (the notebook is modular). 
The code at the end (EvaluationTable) allows evaluation results to be saved for further analysis (see **analysis.ipynb**). 
Evaluation metrics are defined in **evaluation.ipynb**, and the default directory for saving results is **evaluation**.
