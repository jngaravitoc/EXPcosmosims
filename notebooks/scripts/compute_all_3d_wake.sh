# Activate virtual env
sh /home/ngc/Work/research/codes/EXP_tools/activate_env.sh 

halos=("788 756 747 719 666 659 407 349 327 282 229 222 170 169 113 004 983 975")

for halo in $halos
do
    echo $halo 
    python3 3d_density_fields.py ~/Downloads/ ~/Downloads/ halo
done
