import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import libcst as cst
from tqdm import tqdm

from typet5.data import GitRepo, get_dataset_dir
from typet5.type_env import collect_annots_info, mypy_checker
from typet5.utils import proj_root, read_file, write_file, not_none

import pickle

os.chdir(proj_root())
repos_dir = get_dataset_dir("ManyTypes4Py") / "repos"

def count_repo_annots(rep):
    try:
        rep.collect_annotations(repos_dir)
        if rep.n_type_annots / rep.lines_of_code > 0.05:
            return rep
    except Exception as e:
        logging.warning(f"Failed to count annotations for {rep.name}. Exception: {e}")
        return None

def run():
    # download all candidate repos
    all_repos = json.loads(read_file("data/mypy-dependents-by-stars.json"))
    all_repos = [GitRepo.from_json(r) for r in all_repos]
    # all_repos=all_repos[:10] # for testing

    

    def clear_downloaded_repos(repos_dir):
        shutil.rmtree(repos_dir)


    def download_repos(
        to_download: list[GitRepo], repos_dir, download_timeout=10.0, max_workers=10
    ) -> list[GitRepo]:
        def download_single(repo: GitRepo):
            try:
                if repo.download(repos_dir, timeout=download_timeout):
                    repo.read_last_update(repos_dir)
                    return repo
                else:
                    return None
            except subprocess.TimeoutExpired:
                return None
            except Exception as e:
                logging.warning(f"Failed to download {repo.name}. Exception: {e}")
                return None

        print("Downloading repos from Github...")
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fs = [executor.submit(download_single, repo) for repo in to_download]
            rs = [f.result() for f in tqdm(as_completed(fs), total=len(fs))]
        print(f"Downloading took {time.time() - t_start} seconds.")
        downloaded = [r for r in rs if r is not None]
        return downloaded


    if not repos_dir.exists():
        (repos_dir / "downloading").mkdir(parents=True)
        (repos_dir / "downloaded").mkdir(parents=True)
        downloaded_repos = download_repos(all_repos, repos_dir)
        print("Deleting failed repos...")
        shutil.rmtree(repos_dir / "downloading")
    else:
        print("Repos already downloaded.")
        downloaded_dirs = set(d.name for d in (repos_dir / "downloaded").iterdir())
        downloaded_repos = [r for r in all_repos if r.authorname() in downloaded_dirs]
        print("Reading last updates...")
        for r in tqdm(downloaded_repos):
            try:
                r.read_last_update(repos_dir)
            except Exception as e:
                print(r)
                raise e
    print(f"Downloaded {len(downloaded_repos)}/{len(all_repos)} repos.")

    date_threshold = datetime(2021, 4, 20)
    new_repos = [r for r in downloaded_repos if not_none(r.last_update) > date_threshold]
    print(f"{len(new_repos)} / {len(downloaded_repos)} repos are updated within a year.")
    loc_limit = 50000

    small_repos = []
    for rep in tqdm(new_repos):
        try:
            # print(rep)
            loc = rep.count_lines_of_code(repos_dir)
            if loc < loc_limit:
                small_repos.append(rep)
        except UnicodeDecodeError:
            # nothing we can do
            pass
        except Exception as e:
            logging.warning(f"Failed to count lines of code for {rep.name}. Exception: {e}")

    print(
        f"{len(small_repos)}/{len(new_repos)} repos are within the size limit ({loc_limit} LOC)."
    )

    # filter away repos with too few annotations

    


    with ProcessPoolExecutor(max_workers=30) as executor:
        fs = [executor.submit(count_repo_annots, rep) for rep in small_repos]
        rs = [f.result() for f in tqdm(as_completed(fs), total=len(fs))]
    useful_repos: list[GitRepo] = [
        r for r in rs if r is not None and "typeshed" not in r.name
    ]

    print(
        f"{len(useful_repos)}/{len(small_repos)} repos are parsable and have enough portions of type annotations."
    )

    # Some summary statistics

    # print total number of manual annotations
    n_total_annots = sum(not_none(rep.n_type_annots) for rep in useful_repos)
    print("Total number of manual annotations:", n_total_annots)

    # print total number of type places
    n_total_places = sum(not_none(rep.n_type_places) for rep in useful_repos)
    print("Total number of type places:", n_total_places)

    # print total number of lines of code
    n_total_lines = sum(not_none(rep.lines_of_code) for rep in useful_repos)
    print("Total number of lines of code:", n_total_lines)

    # print average number of type annotations per line of code excluding projects with more than 1000 lines of code
    n_avg_annots = (
        sum(not_none(rep.n_type_annots) for rep in useful_repos if rep.lines_of_code < 1000)
        / n_total_lines
    )

    print("Average number of type annotations per line of code:", n_avg_annots)

    useful_repos_path = proj_root() / "scripts" / "useful_repos.pkl"
    with useful_repos_path.open("wb") as f:
        pickle.dump(useful_repos, f)
    print(f"Saved {len(useful_repos)} useful repos to {useful_repos_path}.")
    with useful_repos_path.open("rb") as f:
        print(pickle.load(f)[:3])

    repos_split = pickle_load(Path("data/repos_split.pkl"))
    repos_dir = get_dataset_dir("ManyTypes4Py") / "repos"

    for split, repos in repos_split.items():
        for r in tqdm(repos, desc=f"Moving {split} repos."):
            r: GitRepo
            split: str
            src = repos_dir / "downloaded" / r.authorname()
            (repos_dir / split).mkdir(parents=True, exist_ok=True)
            dest = repos_dir / split / r.authorname()

            if src.exists():
                shutil.move(src, dest)
            else:
                print(f"Repo {r.name} not found.")


if __name__ == "__main__":
    run()