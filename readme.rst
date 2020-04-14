Attribute CNNs für die Einzelworterkennung (Attribute CNNs for Handwriting Word Recognition)
############################################################################################


.. image:: doc/images/pipeline_wrec.png


As part of my bachelors thesis, I implemented experiments that utilized variantions of the PHOCNet for word recognition.
All relevant source code resides in this repository.


All contents of this repository are free for academic use and academic use only.
If you use any of this code for scientific purpose, I would urge you to cite all papers listed in `Related Works`_.
Also, if you heavily base your work on this code I would ask you to consider referencing my thesis listed in `Thesis`_ as well.


.. _`Related Works`:

Related Works
=============

I based my work on the following papers.
Please cite the papers and articles listed below, if you use The PHOCNet, Probabilistic Retrieval Model or residing code.


The PHOCNet by Sudholt and Fink:

::

    @article{SudholtFink:2018IJDAR,
        author = {Sebastian Sudholt and
                  Gernot A. Fink},
        title = {Attribute CNNs for Word Spotting in Handwritten Documents},
        journal = {International Journal on Document Analysis and Recognition (IJDAR)},
        issue_date = {September 2018},
        volume = {21},
        number = {3},
        month = {September},
        year = {2018},
        pages = {199--218},
        numpages = {20},
        publisher = {Springer-Verlag},
        address = {Berlin, Heidelberg}
    }

The Probabilistic Retrieval Model by Rusakov et al.:

::

    @inproceedings{RusakovRMF18,
        author = {Eugen Rusakov and
                  Leonard Rothacker and
                  Hyunho Mo and
                  Gernot A. Fink},
        title = {A Probabilistic Retrieval Model for Word Spotting Based on Direct Attribute Prediction},
        booktitle = {International Conference on Frontiers in Handwriting Recognition (ICFHR)},
        month = {August},
        place = {NY, USA},
        pages = {38--43},
        year = {2018}
    }


Also I took great inspiration from the following Paper by Retsinal et al., when writing this code.
Hence, I would urge you to cite their work as well.

::

    @inproceedings{RetsinasSSLG18,
        author = {George Retsinas and
                  Giorgos Sfikas and
                  Nikolaos Stamatopoulos and
                  Georgios Louloudis and
                  Basilis Gatos},
        title = {Exploring Critical Aspects of CNN-based Keyword Spotting. {A} PHOCNet Study},
        booktitle = {International Workshop on Document Analysis Systems (DAS)},
        pages = {13--18},
        year = {2018}
    }



.. _`Thesis`:

Thesis
======

My thesis is openly available at http://patrec.cs.tu-dortmund.de/pubs/theses/ba_springenberg.pdf.
If you should choose to reference it, I would suggest using the following bibliographical item:

::

    @thesis{ba:TUDo:mspringe,
        author = {Maximilian Springenberg},
        title = {Attribute CNNs für die Einzelworterkennung},
        school = {TU Dortmund},
        year = 2019,
        address = {Dortmund, Germany},
        month = {August},
        note = {(bachelor thesis)}
    }
