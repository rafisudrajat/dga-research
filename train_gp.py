import pandas as pd
import random
from typing import Tuple, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from typing import Any, List, Dict
from sklearn.base import clone

class OneVsRestDataExtraction:
    """
    Utility class to generate training and test data for One-vs-Rest classification from a labeled DataFrame.
    """
    def __init__(self, dataset: Any, selected_class: str, random_state: Optional[int] = None) -> None: 
        self.class_labels = ["D1", "D2", "PD", "T1", "T2", "T3"]
        self.selected_class = selected_class
        self.random_state = random_state
        self._rng = random.Random(random_state)
        self.all_selected_indices = []

        if isinstance(dataset, pd.DataFrame):
            self._dataset = dataset.copy()
            self._use_dataframe = True
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            X, y = dataset
            df = pd.DataFrame(X)
            df["Justifikasi"] = y
            self._dataset = df.copy()
            self._use_dataframe = False
        else:
            raise ValueError("dataset must be a pandas DataFrame or a tuple (X, y) with numpy arrays.")

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset.copy()

    def create_train_data(
        self, 
        selected_class_data_count: int, 
        rest_of_class_data_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data for One-vs-Rest classification.

        Args:
            selected_class_data_count (int): Number of samples to draw from the selected class.
            rest_of_class_data_count (int): Number of samples to draw from each of the other classes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (X, y), where X is the feature matrix and y is the binary label vector.
        """
        class_indices = {
            label: self._dataset.index[self._dataset["Justifikasi"] == label].tolist()
            for label in self.class_labels
        }

        # Sample indices for the selected class
        selected_indices = self._rng.sample(
            class_indices[self.selected_class],
            min(selected_class_data_count, len(class_indices[self.selected_class]))
        )

        # Sample indices for the rest of the classes
        rest_indices = []
        for label in self.class_labels:
            if label == self.selected_class:
                continue
            count = min(rest_of_class_data_count, len(class_indices[label]))
            if count > 0:
                rest_indices.extend(self._rng.sample(class_indices[label], count))

        # Combine and shuffle indices
        all_indices = selected_indices + rest_indices
        self._rng.shuffle(all_indices)
        self.all_selected_indices = all_indices

        # Extract features and labels
        sampled_rows = self._dataset.loc[all_indices]
        X = sampled_rows.drop(columns="Justifikasi").to_numpy()
        y = (sampled_rows["Justifikasi"] == self.selected_class).astype(int).to_numpy()

        return X, y

    def create_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create test data from the remaining samples not used in training.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (X, y), where X is the feature matrix and y is the binary label vector.
        """
        mask = ~self._dataset.index.isin(self.all_selected_indices)
        sampled_rows = self._dataset.loc[mask]

        X = sampled_rows.drop(columns="Justifikasi").to_numpy()
        y = (sampled_rows["Justifikasi"] == self.selected_class).astype(int).to_numpy()

        return X, y
    
class ClassificationReport:
    def __init__(
        self,
        correct_prediction_count: int,
        incorrect_prediction_count: int,
        train_data: np.ndarray,
        train_data_label: np.ndarray,
        test_data: np.ndarray,
        test_data_label: np.ndarray,
        gpc_model: Optional[GaussianProcessClassifier] = None
    ) -> None:
        self.correct_prediction_count = correct_prediction_count
        self.incorrect_prediction_count = incorrect_prediction_count
        self.correct_percentage = correct_prediction_count / (correct_prediction_count+incorrect_prediction_count) * 100
        self.train_data = train_data
        self.train_data_label = train_data_label
        self.test_data = test_data
        self.test_data_label = test_data_label
        self.gpc_model = gpc_model

def train_onevsrest_gpc_with_data_variation(
    gpc_model: GaussianProcessClassifier,
    data_generator: OneVsRestDataExtraction,
    selected_class_data_counts: List[int],
    rest_of_class_data_counts: List[int],
    reports_name: List[str],
    copy_gpc_model: bool = False
) -> Dict[str, ClassificationReport]:
    """
    Trains a GaussianProcessClassifier with varying data sizes and returns performance reports.

    Args:
        gpc_model: The classifier instance.
        data_generator: Data generator for train/test splits.
        selected_class_data_counts: List of positive class sample counts.
        rest_of_class_data_counts: List of negative class sample counts.
        reports_name: List of report names.
        copy_gpc_model: Whether to clone the model for each run.

    Returns:
        Mapping from report name to classification report.
    """
    training_report = {}

    for pos_count, neg_count, report_name in zip(
        selected_class_data_counts, rest_of_class_data_counts, reports_name
    ):
        # Optionally clone the model for each run
        model = clone(gpc_model) if copy_gpc_model else gpc_model

        # Generate training data
        X_train, y_train = data_generator.create_train_data(pos_count, neg_count)
        model.fit(X_train, y_train)

        # Generate test data
        X_test, y_test = data_generator.create_test_data()
        predictions = model.predict(X_test)

        # Calculate correct and incorrect predictions
        correct_pred_count = np.sum(predictions == y_test)
        incorrect_pred_count = len(y_test) - correct_pred_count

        # Store the report
        training_report[report_name] = ClassificationReport(
            correct_prediction_count=correct_pred_count,
            incorrect_prediction_count=incorrect_pred_count,
            train_data=X_train,
            train_data_label=y_train,
            test_data=X_test,
            test_data_label=y_test,
            gpc_model=model if copy_gpc_model else None
        )

    return training_report

def encode_justifikasi(justifikasi:str)->int:
    mapping = {"D1":0,"D2":1,"PD":2,"T1":3,"T2":4,"T3":5}
    return mapping[justifikasi]

def transform_small_dataset(small_dataset:pd.DataFrame)->pd.DataFrame:
    # Fill the NaN or empty value with zero
    small_dataset = small_dataset.fillna(0)
    # Remove last row 
    small_dataset = small_dataset[:-1]

    # Remove unnecessary feature from test data
    # Select the first 5 columns, plus the 9th column (index 8)
    small_dataset = small_dataset.iloc[:, list(range(5)) + [8]]

    return small_dataset

def print_report(reports_name: List[str], reports: Dict[str, ClassificationReport]):
    class_count = 6
    for name in reports_name:
        name_splitted = name.split("_")
        report_name = name_splitted[0]
        selected_class_data_count = int(name_splitted[1])
        rest_of_class_data_count = int(name_splitted[2]) * (class_count - 1)
        print("=" * 40)
        print(f"Classifier: {report_name}")
        print("-" * 40)
        print("Training Data Distribution:")
        print(f"  Selected class samples : {selected_class_data_count}")
        print(f"  Rest of class samples  : {rest_of_class_data_count}")
        print("-" * 40)
        print("Training Accuracy:")
        print(f"  Correct predictions on sampled test data   : {reports[name].correct_prediction_count}")
        print(f"  Incorrect predictions on sampled test data : {reports[name].incorrect_prediction_count}")
        print(f"  Percentage of correct prediction : {reports[name].correct_percentage} %")
        print("=" * 40 + "\n")
        
