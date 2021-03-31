pip3 install -r requirements.txt
pip3 install deepwalk
echo "Deepwalk installed"
pip install node2vec
echo "node2vec installed"

cd dataset/protiens
echo "pair wise node classification on Protiens"
echo "pair wise node classification on Protiens using node2vec started"
python node2vec_pairwise_node_classification_protein.py
echo "pair wise node classification on Protiens using node2vec done"
echo "results at dataset/protiens/result_pairwisenode_protein_node2vec.txt"
echo "pair wise node classification on Protiens using deepwalk started"
python deepwalk_pairwise_node_classification_protein.py
echo "pair wise node classification on Protiens using deepwalk end"
echo "results at dataset/protiens/result_pairwisenode_protein_deepwalk.txt"


cd ../ppi
echo "multi_class_node_classification on PPI"

echo "multi_class_node_classification on PPI using node2Vec started"
python node2vec_multi_class_node_classification_final.py
echo "multi_class_node_classification on PPI using node2Vec done"
echo "results at dataset/ppi/result_multiclass_ppi_node2vec.txt"
echo "multi_class_node_classification on PPI using deepwalk started"
python deepwalk_multi_class_node_classification_final.py
echo "multi_class_node_classification on PPI using deepwalk done"
echo "results at dataset/ppi/result_multiclass_ppi_deepwalk.txt"

echo "link_prediction on PPI"
echo "link_prediction on PPI using node2vec started"
python node2vec_link_pred_ppi_final.py
echo "link_prediction on PPI using node2vec done"
echo "results at dataset/ppi/result_linkpred_ppi_node2vec.txt"
echo "link_prediction on PPI using deepwalk Started"
python deepwalk_link_pred_ppi_final.py
echo "link_prediction on PPI using deepwalk done"
echo "results at dataset/ppi/result_linkpred_ppi_deepwalk.txt"


cd ../Brightkite
echo "link_prediction on Brightkite"
echo "link_prediction on Brightkite using node2vec started"
python mode2vec_link_pred_brightkite_final.py
echo "link_prediction on Brightkite using node2vec done"
echo "results at dataset/Brightkite/result_linkpred_brightkite_node2vec.txt"
echo "link_prediction on Brightkite using deepwalk started"
python deepwalk_link_pred_brightkite_final.py
echo "link_prediction on Brightkite using deepwalk done"
echo "results at dataset/Brightkite/result_linkpred_brightkite_deepwalk.txt"

cd ../..

