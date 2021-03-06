#!/bin/bash

PROG=$1
TYPE=$2

if [ -z $PROG ] || [ $PROG = "-h" ] || [ -z $TYPE ]
then
    echo "Usage: $(basename $0) <executable> <type>" 1>&2
    echo
    echo -e "Types:\tvector, matrix, image" 1>&2
    exit 1
fi

if ! [ -x $PROG ]
then
    echo "File $PROG doesn't exist or isn't executable." 1>&2
    exit 2
fi

DIR=$(readlink -f $(dirname ${PROG}))

DATASET_DIR=${DIR}/datasets/${PROG}-data

if ! [ -d $DATASET_DIR ]
then
    echo "Dataset ${DATASET_DIR} doesn't exist." 1>&2
    exit 3
fi


for i in $(ls ${DATASET_DIR} | sort -h)
do
    if $(${DIR}/run-mp.sh ${PROG} -e $(find ${DATASET_DIR}/${i} | grep output | head -n1) \
                                  -i $(find ${DATASET_DIR}/${i} | grep input | sort -h | paste -d, -s) \
                                  -o /tmp/output.raw -t ${TYPE} \
        | grep -q 'Solution is correct.')
    then
        echo "$(basename ${DATASET_DIR})/${i}: OK"
    else
        echo "$(basename ${DATASET_DIR})/${i}: FAIL"
    fi
done
