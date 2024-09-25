from typet5.utils import *
from typet5.function_dataset import data_project_from_dir
from typet5.static_analysis import SignatureErrorAnalysis, AccuracyMetric
from typet5.model import ModelWrapper
from typet5.experiments.typet5 import TypeT5Configs
from typet5.utils import pickle_load, Path, proj_root, tqdm
import shutil
from typet5.data import GitRepo, get_dataset_dir
from collections import Counter

# load projects
# dataset_name = "InferTypes4Py"
dataset_name = "ManyTypes4Py"
split = "test"
repos_dir = get_dataset_dir(dataset_name) / "repos" / split

# parse all type annotations
def get_repo_annots(repo: GitRepo):
    annots = []
    file_to_annots = repo.collect_annotations(repos_dir, silent=True, without_downloaded=True)
    # print(file_to_annots)
    for d in file_to_annots.values():
        annots.extend(d.values())
    return annots

def run():
    os.chdir(proj_root())
    
    
    useful_repos = pickle_load(Path("scripts/useful_repos.pkl"))

    # print(useful_repos)
    # exit()

    
    repo_paths = [f for f in repos_dir.iterdir() if f.is_dir()]
    projects = pmap(
        data_project_from_dir,
        repo_paths,
        desc="Loading projects",
    )

    labels = {p.name: p.get_sigmap() for p in projects}
    model_path = get_model_dir() / TypeT5Configs.Default.get_model_name()
    # common_names can be obtained in other ways, but here we just load it from the model dir
    common_names = ModelWrapper.load_common_type_names(model_path)
    metric = AccuracyMetric(common_names, relaxed_equality=False, filter_none_any=False, name="full_acc")

    print("Type slots:", sum(sig.n_annots() for p in projects for sig in p.get_sigmap().values()))
    pretty_print_dict(SignatureErrorAnalysis(labels, labels, metric).accuracies)



    with ProcessPoolExecutor(max_workers=20) as executor:
        all_annots = list(seq_flatten(pmap(get_repo_annots, useful_repos, desc="Parsing annotations")))

    n_total = len(all_annots)
    all_annots = [t for t in all_annots if t[0] is not None]
    n_failed_to_parse = n_total - len(all_annots)
    print("total number of parsed annotations: ", len(all_annots))
    print(f"{n_failed_to_parse / n_total * 100:.3f}% failed to parse.")

    def freq_table(labels, at_least=0):
        counts = Counter(labels).most_common(None)
        rows = filter(lambda row: row[1] >= at_least, counts)
        return pd.DataFrame(rows, columns=["label", "count"])

    def plot_counts(labels, title:str="", top_k=100):
        freq = freq_table(labels)
        subset = freq[0:top_k]
        coverage = sum(subset["count"]) / sum(freq["count"])
        print(f"Number of different labels: {len(freq)}")
        print(f"Top-{top_k} labels achieve coverage: {coverage * 100:.2f}%")
        # data = pd.DataFrame(subset)
        # fig = px.bar(data, x="label", y="count", title=title)
        # fig.update_xaxes(visible=True, showticklabels=False)
        # return fig

    def type_name(xs):
        if xs == ():
            return "[empty]"
        else:
            return xs[-1]

    # for a in all_annots:
    #     print(a)
    #     input()
    #     print(a[0])
    #     print(a[1])
    #     print(a[0].head)
    #     print(a[1].head)
    # freq_table(all_heads, at_least=3)

    plot_counts((map(str,all_annots)), title="Full Types")
    plot_counts((map(lambda a: type_name(a[0].head), all_annots)), title="Partial Types")
    all_heads = [type_name(h) for t in all_annots for h in t[0].all_heads()]
    plot_counts(all_heads, title="Type Constructor Names")

    print(freq_table(all_heads, at_least=3))

if __name__ == "__main__":
    run()