import pathlib
import stat
import sys
import zipfile


EXPECTED = {
    "IndexAdd": (
        "op_host/index_add.cpp",
        "op_host/index_add_tiling.h",
        "op_host/aclnn_index_add_compat.cpp",
        "op_kernel/index_add.cpp",
    ),
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
        "op_host/aclnn_square_sum_v1_compat.cpp",
        "op_kernel/square_sum_v1.cpp",
    ),
}


def validate(archive):
    op = archive.stem
    if op not in EXPECTED:
        raise SystemExit(f"unexpected archive name: {archive.name}")
    prefix = f"{op}_zip/"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        required_dirs = {
            prefix,
            prefix + "op_host/",
            prefix + "op_kernel/",
        }
        if not required_dirs.issubset(names):
            missing = sorted(required_dirs.difference(names))
            raise SystemExit(f"{archive.name}: missing directories {missing}")
        for info in zf.infolist():
            name = info.filename
            if "\\" in name or not name.startswith(prefix):
                raise SystemExit(f"{archive.name}: invalid entry {name}")
            if info.create_system != 3:
                raise SystemExit(f"{archive.name}: non-Unix metadata for {name}")
            suffix = name[len(prefix):]
            allowed = (
                suffix == ""
                or suffix == "op_host/"
                or suffix == "op_kernel/"
                or suffix.startswith("op_host/")
                or suffix.startswith("op_kernel/")
                or suffix == "custom_opp_ubuntu_aarch64.run"
            )
            if not allowed:
                raise SystemExit(f"{archive.name}: unrelated entry {name}")
        for suffix in EXPECTED[op]:
            if prefix + suffix not in names:
                raise SystemExit(f"{archive.name}: missing {suffix}")
        runs = [name for name in names if name.endswith(".run")]
        expected_run = prefix + "custom_opp_ubuntu_aarch64.run"
        if runs != [expected_run]:
            raise SystemExit(
                f"{archive.name}: run package must be exactly {expected_run}; got {runs}"
            )
        run_mode = zf.getinfo(expected_run).external_attr >> 16
        if not run_mode & stat.S_IXUSR:
            raise SystemExit(f"{archive.name}: .run is not executable")


def main(root):
    root = pathlib.Path(root)
    archives = sorted(root.glob("*.zip"))
    if [path.stem for path in archives] != sorted(EXPECTED):
        raise SystemExit(
            f"expected {sorted(EXPECTED)}, got {[path.stem for path in archives]}"
        )
    for archive in archives:
        validate(archive)
        print(f"PASS {archive.name}")


if __name__ == "__main__":
    main(sys.argv[1])
