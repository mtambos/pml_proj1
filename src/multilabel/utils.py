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

HEATMAP_CMAP = ListedColormap(
    sns.color_palette('rocket', n_colors=100).as_hex()
)
DATA_COLORS = (
    (0.29803921568627451, 0.44705882352941179, 0.69019607843137254),
    (0.33333333333333331, 0.6588235294117647, 0.40784313725490196),
    (0.7686274509803922, 0.30588235294117649, 0.32156862745098042),
    (0.80000000000000004, 0.72549019607843135, 0.45490196078431372),
    (0.39215686274509803, 0.70980392156862748, 0.80392156862745101),
    (0.50588235294117645, 0.44705882352941179, 0.69803921568627447),
)
DATA_CMAP = ListedColormap(DATA_COLORS)
DATA_MARKERS = ('^', 'v', 's', 'd')


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

    X = np.r_[X_train, X_test]
    x_min, x_max = X[:, 0].min() * 1.1, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 1.1, X[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=DATA_CMAP)

    ax.scatter(*X_train[y_train == -1].T, color=DATA_COLORS[0],
               edgecolors='black', marker=DATA_MARKERS[0], s=50,
               label='-train')
    ax.scatter(*X_train[y_train == 1].T, color=DATA_COLORS[1],
               edgecolors='black', marker=DATA_MARKERS[1], s=50,
               label='+train')
    ax.scatter(*X_test[y_test == -1].T, color=DATA_COLORS[0],
               edgecolors='black', marker=DATA_MARKERS[2], s=50,
               label='-test')
    ax.scatter(*X_test[y_test == 1].T, color=DATA_COLORS[1],
               edgecolors='black', marker=DATA_MARKERS[3], s=50,
               label='+test')
    ax.grid()
    ax.legend()


def plot_kernel_importances(e_mu_orig, kernel_attrs, colormap=HEATMAP_CMAP):
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
    plt.suptitle(r'Kernel importance plot ($e_\mu$)')


def plot_e_a_mu(model1, model2, model1_name, model2_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax1, ax2 = axes
    model1.plot_e(ax=ax1, rug=True, rug_kws={'color': 'r', 'linewidth': 2,
                                             'alpha': 1})
    model2.plot_e(ax=ax2, rug=True, rug_kws={'color': 'r', 'linewidth': 2,
                                             'alpha': 1})
    ax1.set_title(model1_name)
    ax2.set_title(model2_name)
    fig.suptitle(r'$e_\mu$ distplot')
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax1, ax2 = axes
    model1.plot_a(ax=ax1, rug=True, rug_kws={'color': 'r', 'linewidth': 2,
                                             'alpha': 1})
    model2.plot_a(ax=ax2, rug=True, rug_kws={'color': 'r', 'linewidth': 2,
                                             'alpha': 1})
    ax1.set_title(model1_name)
    ax2.set_title(model2_name)
    fig.suptitle(r'$a_\mu$ distplot')
    plt.tight_layout()


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
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    return


def plot_classification_report(y_test, y_pred, ax=None, cbar=True,
                               cbar_ax=None, cbar_orient='vertical'):
    assert cbar_orient in ('horizontal', 'vertical')
    if ax is None:
        ax = plt.figure(figsize=(8, 8)).gca()

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_test))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_test, y_pred)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep[:, :-1],
                annot=True,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax,
                vmin=0,
                vmax=1,
                cbar=cbar,
                cbar_ax=cbar_ax,
                cbar_kws={"orientation": cbar_orient})

    support = np.zeros_like(rep) * np.nan
    support[:, -1] = rep[:, -1]
    sns.heatmap(support,
                annot=True,
                cbar=False,
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax)

    return


def plot_confusion_matrix(y_test, y_pred, cmap=HEATMAP_CMAP, ax=None,
                          cbar=True, cbar_ax=None, cbar_orient='vertical'):
    assert cbar_orient in ('horizontal', 'vertical')
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / cm.sum(axis=1, keepdims=True)

    classes = [-1, 1]
    if ax is None:
        ax = plt.figure(figsize=(8, 8)).gca()

    sns.heatmap(cm, ax=ax, annot=True,
                xticklabels=classes, yticklabels=classes,
                cbar=cbar,
                cbar_ax=cbar_ax,
                cbar_kws={"orientation": cbar_orient},
                vmin=0, vmax=1)

    ax.set_ylabel('True label')
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.set_xlabel('Predicted label')
    ax.xaxis.set_label_coords(0.5, -0.03)
    return


def plot_metrics(y_true, y_pred, y_scores, axes=None, cbar_orient='vertical'):
    if axes is None:
        _, axes = plt.subplots(3, 1)
    ax1, ax2, ax3 = axes.flatten()
    plot_classification_report(y_true, y_pred, ax=ax1, cbar_orient=cbar_orient)
    plot_confusion_matrix(y_true, y_pred, ax=ax2, cbar_orient=cbar_orient)
    plot_rocauc(y_true, y_scores, ax=ax3)


def evaluate_model(model, X, y, kernel_attrs, cmap=HEATMAP_CMAP):
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2, stratify=y)

    model.fit(X_train, y_train)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    model.plot_bounds(ax=ax)
    sns.despine()
    plt.tight_layout()

    plot_kernel_importances(model.e_mu_orig, kernel_attrs, cmap)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    model.plot_e(rug=True, ax=ax,
                 rug_kws={'color': 'r', 'linewidth': 2,
                          'alpha': 1})
    ax.set_title(r'$e_\mu$ distplot')
    plt.tight_layout()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    model.plot_a(rug=True, ax=ax,
                 rug_kws={'color': 'r', 'linewidth': 2,
                          'alpha': 1})
    ax.set_title(r'$a_\mu$ distplot')
    plt.tight_layout()

    y_pred = model.predict(X_test)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_classification_report(y_test, y_pred, ax=ax)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_confusion_matrix(y_test, y_pred, cmap=cmap, ax=ax,
                          cbar_orient='vertical')

    y_score = model.predict_proba(X_test)[:, 1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_rocauc(y_test, y_score, ax=ax)
    sns.despine()
    plt.tight_layout()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plot_decision_boundaries(model, X_train, y_train, X_test, y_test, ax=ax)
    sns.despine()
    plt.tight_layout()

    model_cv = clone(model)
    scoring.iteration = 0
    folds = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)
    return cross_validate(model_cv, X, y, cv=folds, scoring=scoring)
