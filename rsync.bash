#!/bin/bash

SERVER=codon-login
DIR=/hps/nobackup/arl/chembl/lorenzo/kalasanty/model_trained
rsync -avz -e ssh ${SERVER}:${DIR} .
#--include "predictions/*"
#--include "*best_checkpoint.pytorch"
