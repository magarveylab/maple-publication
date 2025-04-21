import logging
import os
import sys

# set logging
logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, stream=sys.stdout
)


def run_qtofpeakpicker(
    wiff_dir,
    mzxml_dir,
    wiff_fh,
    mzxml_fh,
    resolution=60000,
    threshold=50,
    backup_dir="backup",
):
    logging.info("Converting wiff files to mzxml")
    # set up directories
    os.makedirs(mzxml_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    # check for wiff scan file
    wiff_scan_fp = "{}/{}.scan".format(wiff_dir, wiff_fh)
    if os.path.exists(wiff_scan_fp) == False:
        return {"status": "Missing wiff scan file", "mzxml_fp": None}
    # delete temp files
    current_mzxml_fp = "{}/{}".format(wiff_dir, mzxml_fh)
    if os.path.exists(current_mzxml_fp):
        os.system("rm -rf {}".format(current_mzxml_fp))
    # check if mzxml previously created
    new_mzxml_fp = "{}/{}".format(mzxml_dir, mzxml_fh)
    if os.path.exists(new_mzxml_fp) == True:
        return {"status": "mzXML already present", "mzxml_fp": new_mzxml_fp}
    # create a backup of wiff file
    current_wiff_fp = "{}/{}".format(wiff_dir, wiff_fh)
    backup_wiff_fp = "{}/{}".format(backup_dir, wiff_fh)
    if os.path.exists(backup_wiff_fp) == False:
        os.system("cp -r {} {}".format(current_wiff_fp, backup_wiff_fp))
    # format docker command
    root = wiff_dir.split("/")[-1]
    docker_cmd = "docker run {opt} -v {m}:/{r} {l} qtofpeakpicker {s} -I /{r}/{i} -O /{r}/{o}"
    license_fp = "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine"
    settings = "--resolution={} --threshold={}".format(resolution, threshold)
    opt = "-it --rm -e WINEDEBUG=-all"
    docker_cmd = docker_cmd.format(
        opt=opt,
        m=wiff_dir,
        r=root,
        l=license_fp,
        s=settings,
        i=wiff_fh,
        o=mzxml_fh,
    )
    # run command
    os.system(docker_cmd)
    # transfer mzxml to final directory
    os.system("mv {} {}".format(current_mzxml_fp, new_mzxml_fp))
    # transfer wiff file
    os.system("mv {} {}".format(backup_wiff_fp, current_wiff_fp))
    # cleanup directory
    os.system("rm -rf {}".format(current_mzxml_fp))
    return {"status": "Complete", "mzxml_fp": new_mzxml_fp}
