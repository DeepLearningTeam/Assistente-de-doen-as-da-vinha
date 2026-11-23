# RAGAs e a avaliação de respostas

Es, James, Espinosa Anke e Schockaert (2024) propõem medir separadamente a relevância da recuperação, a fidelidade da geração ao contexto e a relevância da resposta à pergunta. Para a fidelidade, decompõem a resposta em afirmações e verificam se cada uma pode ser inferida do contexto. A proposta usa modelos para automatizar a avaliação. [Artigo na ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/)

No Tema F, a ideia de decompor a resposta permite que dois anotadores revejam os mesmos trechos. O Recall de página@5 mostra se a página de referência foi recuperada. A fidelidade manual mostra se a resposta realmente usou evidência adequada. Um acerto na primeira medida pode coexistir com uma afirmação sem suporte na segunda.
