# Databricks notebook source
import tempfile
import os
import shutil
import pytube.cipher as CC
fname = os.path.abspath(CC.__file__)

# Patch Based upon: https://stackoverflow.com/questions/70776558/pytube-exceptions-regexmatcherror-init-could-not-find-match-for-w-w

with tempfile.TemporaryDirectory() as tmp_dir:
    out_file = f"{tmp_dir}/test.py"
    with open(fname, "r") as tt, open(out_file, "w") as oo:
        for i, line in enumerate(tt.readlines()):
            if i == 29:
                oo.write('        var_regex = re.compile(r"^\$*\w+\W")\n')
            else:
                oo.write(line)

    shutil.copyfile(out_file, fname )

# COMMAND ----------

dbutils.library.restartPython()

