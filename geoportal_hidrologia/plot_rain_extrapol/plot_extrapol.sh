#!/bin/bash

date
appdir=`dirname $0`
logfile=$appdir/plot_extrapol.log
lockfile=$appdir/plot_extrapol.lck
pid=$$

echo $appdir

function plot_extrapol {

python /media/nicolas/Home/Jupyter/Soraya/git/Alarmas/06_Crones/plot_extrapol.py

}


(
        if flock -n 301; then
                cd $appdir
                plot_extrapol
                echo $appdir $lockfile
                rm -f $lockfile
        else
            	echo "`date` [$pid] - Script is already executing. Exiting now." >> $logfile
        fi
) 301>$lockfile

exit 0