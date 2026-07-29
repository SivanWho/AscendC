import pathlib
import stat
import sys
import zipfile


EXPECTED = {
    "Concat": (
        "op_host/concat.cpp",
        "op_host/concat_tiling.h",
        "op_kernel/concat.cpp",
    ),
    "Greater": (
        "op_host/greater.cpp",
        "op_host/greater_tiling.h",
        "op_kernel/greater.cpp",
    ),
    "SquareSumV1": (
        "op_host/square_sum_v1.cpp",
        "op_host/square_sum_v1_tiling.h",
        "op_kernel/square_sum_v1.cpp",
    ),
}


def main(root):
    root = pathlib.Path(root)
    archives = sorted(root.glob("*/zipfiles/*.zip"))
    if len(archives) != 6:
        raise SystemExit(f"expected six archives, got {len(archives)}")

    for archive in archives:
        op = archive.stem.removesuffix("_benchmark")
        prefix = f"{op}_zip/"
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if not names or any("\\" in name for name in names):
                raise SystemExit(f"invalid path separator in {archive}")
            required_dirs = {
                prefix,
                prefix + "op_host/",
                prefix + "op_kernel/",
            }
            if not required_dirs.issubset(names):
                missing = sorted(required_dirs.difference(names))
                raise SystemExit(f"missing explicit directories {missing} in {archive}")
            for info in zf.infolist():
                if info.create_system != 3:
                    raise SystemExit(f"non-Unix ZIP metadata for {info.filename} in {archive}")
                if not info.filename.startswith(prefix):
                    raise SystemExit(f"unexpected archive entry {info.filename} in {archive}")
            for suffix in EXPECTED[op]:
                required = prefix + suffix
                if required not in names:
                    raise SystemExit(f"missing {required} in {archive}")
            runs = [name for name in names if name.startswith(prefix) and name.endswith(".run")]
            if len(runs) != 1:
                raise SystemExit(f"expected one run package in {archive}, got {runs}")
            expected_run = prefix + "custom_opp_ubuntu_aarch64.run"
            if runs[0] != expected_run:
                raise SystemExit(f"unexpected run name {runs[0]} in {archive}; expected {expected_run}")
            mode = zf.getinfo(runs[0]).external_attr >> 16
            if not mode & stat.S_IXUSR:
                raise SystemExit(f"run package is not executable in {archive}")
        print(f"PASS {archive.name}")


if __name__ == "__main__":
    main(sys.argv[1])
