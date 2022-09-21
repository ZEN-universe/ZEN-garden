#!/bin/bash

for d in test_*; do
  num=${d##test_}
  echo $num
  echo $d
  # mv the files
  git mv ${d}/config.py ${d}/config_${num}.py
  git mv ${d}/system.py ${d}/system_${num}.py

  # some replace and delete
  sed -i "s/from system import system/from system_${num} import system/" ${d}/config_${num}.py
  sed -i "s/from   config                                    import config/from   config_${num}                                 import config/" ${d}/${d}.py
  sed -i '/\@pytest\.mark\.forked/d' ${d}/${d}.py
  sed -i '/\# wrap in function for pytest/d' ${d}/${d}.py
  sed -i "s/modelName \= config\.analysis\[\"dataset\"\]/nameDir \= os\.path\.join\(config\.analysis\[\"dataset\"\]\, \"outputs\"\)/" ${d}/${d}.py
  sed -i 's/modelName=modelName/nameDir=nameDir/' ${d}/${d}.py
  sed -i 's/modelName/nameDir/' ${d}/${d}.py
  sed -i 's/from   zen_garden.preprocess.prepare             import Prepare/from   zen_garden                                import restore_default_state\nfrom   zen_garden.preprocess.prepare             import Prepare/' ${d}/${d}.py
  sed -i 's/\# reset the energy system/\# restore defaults/' ${d}/${d}.py
  sed -i 's/EnergySystem\.reset_system/restore_default_state/' ${d}/${d}.py
done
