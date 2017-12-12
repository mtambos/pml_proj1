import itertools

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, auc, roc_curve, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from yellowbrick.classifier import (ClassificationReport, ROCAUC,
                                    ConfusionMatrix,
                                    DecisionBoundariesVisualizer)

heatmap_cmap = ListedColormap(sns.color_palette('rocket').as_hex())


def matlab2np(mlab_array):
    return np.array(
        list(
            map(
                lambda x: list(map(float, x.split())),
                mlab_array.split('\n')
            )
        )
    )


# noinspection PyPep8Naming
def poly_kernel(A, B, c, d):
    return (A @ B.T + c)**d


# noinspection PyPep8Naming
def gauss_kernel(A, B, sigma):
    return np.exp(-cdist(A, B, metric='sqeuclidean')/sigma)


# noinspection PyPep8Naming
def scoring(estimator, X_test, y_test):
    if 'iteration' not in dir(scoring):
        scoring.iteration = 0
    y_pred = estimator.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    bemkl_model = estimator
    if isinstance(estimator, Pipeline):
        bemkl_model = estimator.named_steps['bemkl']
    e_mu = bemkl_model.b_e_mu[0, 1:]
    X_train = bemkl_model.X_train
    if len(X_train) != len(X_test):
        print(
            f"{scoring.iteration} - "
            f"Kernels: {bemkl_model.nr_kernels_used}/"
            f"{bemkl_model.total_kernels} "
            f"({bemkl_model.nr_kernels_used/bemkl_model.total_kernels}). "
            f"SV: {bemkl_model.nr_sv_used}/{bemkl_model.total_sv} "
            f"({bemkl_model.nr_sv_used/bemkl_model.total_sv}). "
            f"Mean e: {e_mu.mean():0.4f}. "
            f"Median e: {np.median(e_mu):0.4f}. "
            f"Std e: {e_mu.std():0.4f}. "
         )
        scoring.iteration += 1
    return score


def plot_decision_boundaries(model, X_train, y_train,
                             X_test, y_test, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(8, 8)).gca()
    dv = DecisionBoundariesVisualizer(model, classes=[0, 1, 2, 3])
    markers = (next(dv.markers), next(dv.markers),
               next(dv.markers), next(dv.markers))
    colors = dv.colors
    dv = DecisionBoundariesVisualizer(model, classes=[-1, 1],
                                      ax=ax, step_size=0.025,
                                      show_scatter=False,
                                      pcolormesh_alpha=0.25)

    ax.scatter(*X_train[y_train == -1].T, color=colors[0],
               edgecolors='black', marker=markers[0], s=30,
               label='-train')
    ax.scatter(*X_train[y_train == 1].T, color=colors[1],
               edgecolors='black', marker=markers[1], s=30,
               label='+train')
    ax.scatter(*X_test[y_test == -1].T, color=colors[2],
               edgecolors='black', marker=markers[2], s=30,
               label='-test')
    ax.scatter(*X_test[y_test == 1].T, color=colors[3],
               edgecolors='black', marker=markers[3], s=30,
               label='+test')
    dv.fit_draw_poof(X_test, y_test)
    ax.legend()


def plot_kernel_importances(e_mu_orig, kernel_attrs, colormap=heatmap_cmap):
    kernel_attrs = np.asarray(kernel_attrs)
    df = pd.DataFrame(np.c_[e_mu_orig, kernel_attrs],
                      columns=['e_mu', 'kernel', 'features', 'parameter'])\
           .astype({'e_mu': float, 'kernel': str,
                    'features': str, 'parameter': float})

    kernel_nr = len(set(df.kernel))
    fig, axes = plt.subplots(1, kernel_nr, figsize=(8*kernel_nr, 8))
    axes = axes.flatten()
    vmin, vmax = df.e_mu.min(), df.e_mu.max()
    for ax, (kernel, group) in zip(axes, df.groupby('kernel')):
        group = group.pivot("parameter", "features", "e_mu")
        sns.heatmap(group, vmin=vmin, vmax=vmax, cmap=colormap, ax=ax)
        ax.set_title(kernel)


def evaluate_model(model, X, y, kernel_attrs, cmap=heatmap_cmap):
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, stratify=y)

    model.fit(X_train, y_train)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    model.plot_bounds(ax=ax)

    plot_kernel_importances(model.e_mu_orig, kernel_attrs, cmap)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    model.plot_e(rug=True, ax=ax,
                 rug_kws={'color': 'r', 'linewidth': 2,
                          'alpha': 1})
    ax.set_title(r'$e_\mu$ distplot')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    model.plot_a(rug=True, ax=ax,
                 rug_kws={'color': 'r', 'linewidth': 2,
                          'alpha': 1})
    ax.set_title(r'$a_\mu$ distplot')

    y_pred = model.predict(X_test)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_classification_report(y_test, y_pred, ax=ax)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_confusion_matrix(y_test, y_pred, cmap=cmap, ax=ax)

    y_score = model.predict_proba(X_test)[:, 1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_rocauc(y_test, y_score, ax=ax)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_decision_boundaries(model, X_train, y_train, X_test, y_test, ax=ax)

    model_cv = clone(model)
    scoring.iteration = 0
    folds = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)
    return cross_validate(model_cv, X, y, cv=folds, scoring=scoring)


def plot_classification_report(y_test, y_pred, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(8, 8)).gca()

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_test))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_test, y_pred)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True,
                cbar=False,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax, cmap=heatmap_cmap)
    return


def plot_rocauc(y_test, y_score, ax=None):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        ax = plt.figure(figsize=(8, 8)).gca()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")
    return


def plot_confusion_matrix(y_test, y_pred, cmap=heatmap_cmap, ax=None):
    """
    This function prints and plots the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = [-1, 1]
    if ax is None:
        ax = plt.figure(figsize=(8, 8)).gca()

    ax_pos = ax.get_position()
    cbar_pos = [
        ax_pos.x0 + ax_pos.width + 0.023,
        ax_pos.y0 + ax_pos.height*0.05,
        ax_pos.width*0.04,
        ax_pos.height - ax_pos.height*0.1,
    ]
    fig = ax.figure
    cax = fig.add_axes(cbar_pos)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.tick_params(labelsize='xx-small', )

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f'
    thresh = cm.max() - (cm.max() - cm.min()) * 0.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] < thresh else "black")

    ax.set_ylabel('True label')
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.set_xlabel('Predicted label')
    ax.xaxis.set_label_coords(0.5, -0.03)
    return


def plot_metrics(y_true, y_pred, y_scores, axes=None):
    if axes is None:
        _, axes = plt.subplots(3, 1)
    ax1, ax2, ax3 = axes.flatten()
    plot_classification_report(y_true, y_pred, ax=ax1)
    plot_confusion_matrix(y_true, y_pred, ax=ax2)
    plot_rocauc(y_true, y_scores, ax=ax3)
