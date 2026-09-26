# SINANBIO · IA para descoberta de ativos da Amazônia

Oito simulações executáveis que acompanham a palestra **"Inteligência artificial como propulsora na descoberta de ativos da Amazônia"** (Sinanbio, 2026).

> ## ⚠️ Todos os dados deste repositório são sintéticos
>
> Nenhum arquivo aqui contém molécula real, espectro medido, genoma sequenciado, ensaio biológico, espécie nomeada, comunidade identificada ou registro administrativo verdadeiro. Tudo é gerado por simulação, com semente fixa, para tornar **mecanismos** visíveis.
>
> Isso não é uma limitação disfarçada de virtude. É o contrário: a confusão entre predição computacional e descoberta confirmada é o vício mais comum da literatura de "IA para produtos naturais", e um repositório que ilustra esse campo não pode cometer o mesmo erro. Cada demo declara no cabeçalho o seu nível de evidência, e cada figura sai carimbada.

---

## Por que este repositório existe

Uma palestra de 25 minutos cabe em slides. O que não cabe é a verificação: o ouvinte não tem como checar, ali, se um número é robusto ou se é artefato de avaliação. Este repositório é a contrapartida — o código roda em um minuto e meio, em qualquer máquina, e cada afirmação da palestra que depende de um mecanismo computacional tem aqui a sua demonstração reprodutível.

A tese que o conjunto sustenta é simples e desconfortável: **na maior parte do caminho, o gargalo da IA para a Amazônia não é o algoritmo. É o dado, o desenho experimental e a governança.** Os demos 01, 06 e 07 mostram isso de três ângulos independentes.

---

## Escala de níveis de evidência

Usada em todos os cabeçalhos, no terminal e neste README.

| Nível | Significado |
|---|---|
| **E1** | Validação clínica ou produto aprovado |
| **E2** | Validação experimental *in vivo* |
| **E3** | Validação *in vitro*, ou isolamento e elucidação estrutural confirmados |
| **E4** | Predição computacional com validação retrospectiva ou benchmark |
| **E5** | Predição computacional sem validação, proposta teórica ou revisão narrativa |

**Os resultados deste repositório são E5 por construção** — são simulações. Alguns demonstram fenômenos que são E4/E3 na literatura (indicado caso a caso). Nenhum resultado aqui é descoberta.

---

## Os oito demos

Cada script imprime um relatório no terminal, salva uma figura em `figuras/` com manifesto de procedência em JSON e exporta a tabela subjacente em `dados_sinteticos/`.

### `01_lacuna_amazonica.py` — A Amazônia é um problema de dados
Um modelo só enxerga o espaço químico do seu treino. Na simulação, **61% das moléculas "amazônicas" caem fora do domínio de aplicabilidade** do acervo global (Tanimoto ao vizinho mais próximo < 0,40). Extrapolação não é erro do modelo; é ausência de dado — e a curva de depósito mostra a cobertura saltando de 46% para 93% com 100 moléculas depositadas. *Nível: E5 (ilustração de mecanismo).*

### `02_divisao_por_scaffold.py` — O número que engana
O mesmo modelo, as mesmas moléculas, duas partições. Divisão aleatória: **AUC 0,936**. Divisão por arcabouço: **AUC 0,732**, precisão média despencando de 0,76 para 0,33. A causa está no primeiro painel: análogos da mesma família têm Tanimoto médio 0,45 contra 0,07 entre famílias. Este é o **teste mínimo de honestidade** a exigir de qualquer resultado apresentado como descoberta. *Nível: E4 (fenômeno documentado em benchmarks de quimioinformática; a demonstração é sintética).*

### `03_redes_moleculares.py` — O gargalo é anotar, não medir
Com uma biblioteca espectral enviesada (48% de anotação nas famílias bem estudadas, 2% nas tropicais), **100% dos agrupamentos órfãos da rede são compostos por famílias sub-representadas**. O viés da base vira topologia — e vira, ao mesmo tempo, uma lista de prioridade de isolamento. A propagação de anotação leva a cobertura de 28% para 68%, mas acerta 54%: propagar não é identificar. O terceiro painel mostra que o limiar de cosseno move cobertura e acerto em direções opostas — é política de risco, não detalhe técnico. *Nível: E5.*

### `04_genome_mining.py` — A regra encontra o que foi escrita para ver
Detector por regra especialista: **recall 1,00 em clusters modulares e 0,00 em crípticos**. Modelo aprendido: 1,00 e 0,34, ao custo do dobro de falsos positivos. O detalhe que mais importa: o *recall global* da regra é 0,83, porque clusters modulares são mais longos e dominam a contagem — **a métrica agregada apaga por completo o modo de falha**. *Nível: E5 (a assimetria é E4/E3 na literatura de mineração genômica).*

### `05_aprendizado_ativo.py` — A IA não descobre, ela decide o que testar
Com 20% da quimioteca ensaiada, aprendizado ativo encontra **3,2× mais ativos** que triagem aleatória. Mas a quimioteca tem dois farmacóforos independentes, e o ganho inteiro foi gasto aprofundando o primeiro: **triplicar os acertos não comprou uma única molécula a mais do mecanismo raro.** Nenhuma das políticas informadas superou o acaso nessa métrica — nem somar incerteza, nem penalizar redundância, nem reservar cota de exploração. Se o objetivo é *classe nova*, ele precisa estar escrito na função objetivo; o otimizador entrega exatamente o que foi pedido. *Nível: E5 (o ganho do aprendizado ativo é E4/E2 conforme a plataforma).*

### `06_bhe_calibracao.py` — A ponte com a neurociência, e onde ela range
Preditor de permeabilidade da barreira hematoencefálica treinado em espaço "tipo fármaco" e testado em moléculas "tipo produto natural": acurácia cai de **0,837 para 0,606** e o erro de calibração sobe de **0,023 para 0,138**. O modelo erra *com convicção* — a lacuna entre confiança declarada e acurácia é de +0,10. E abster-se pelas predições menos confiantes não salva: recusar metade dos casos leva a acurácia de 0,606 para apenas 0,660, porque a abstenção se apoia justamente na confiança que quebrou. Há ainda um teto irredutível (0,774) imposto pelo efluxo, uma variável latente que nenhum descritor fisico-químico captura. *Nível: E5 (as relações TPSA/massa/lipofilia–BHE e o papel do efluxo são E2/E3 na literatura).*

### `07_dinamica_biossintese.py` — Descobrir a equação, não só o padrão
Identificação esparsa (SINDy, implementado em ~15 linhas) aplicada a uma rota biossintética simulada. Com **uma** curva temporal densa e limpa o método falha catastroficamente (erro médio de coeficiente 12,3). Com **cinco doses de elicitação** ele recupera o sistema exatamente (erro 0,0002). A falha não é do algoritmo: uma trajetória única deixa a matriz da biblioteca mal condicionada. **É uma prescrição de delineamento experimental derivada da matemática.** *Nível: E5 — a aplicação a rotas biossintéticas de produtos naturais é proposta nossa, não literatura estabelecida. O método é E1/E2 em física e engenharia.*

### `08_procedencia_auditavel.py` — Procedência como código
Livro-razão sintético de 240 acessos, cinco regras de auditoria, e a constatação central: dos 93 acessos que entraram em treino de modelo, **apenas 31% são plenamente rastreáveis**. O resto não é necessariamente ilícito — é *inverificável*, que para efeito de confiança científica dá quase no mesmo. A procedência não se perde por má-fé; ela evapora por atrito, um elo de cada vez. A função `auditar()` é o entregável reutilizável: troque as colunas pelos campos do seu laboratório. *Nível: E5. **As regras são caricatura didática e não são assessoria jurídica.***

---

## Como rodar

```bash
git clone https://github.com/<ORG-DO-LAB>/sinanbio-ia-amazonia.git
cd sinanbio-ia-amazonia
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py          # todos os demos (~1,5 min)
python run_all.py 02 07    # apenas os demos 02 e 07
python demos/02_divisao_por_scaffold.py
```

Python 3.11+. Sem GPU, sem download, sem chave de API, sem acesso à rede. As dependências são apenas NumPy, pandas, SciPy, scikit-learn, Matplotlib e NetworkX.

## Estrutura

```
sinanbio-ia-amazonia/
├── demos/               # os oito scripts, autocontidos e comentados
├── src/sinanbio/
│   ├── style.py         # paleta acessível, exportação com manifesto de procedência
│   └── synth.py         # geradores sintéticos e similaridade de Tanimoto
├── figuras/             # PNG 300 dpi + <nome>.manifesto.json por figura
├── dados_sinteticos/    # CSV com os dados que produziram cada figura
├── run_all.py
├── requirements.txt
├── CITATION.cff
└── LICENSE              # MIT
```

## Reprodutibilidade

Toda semente aleatória está fixada (`SEMENTE = 20260926`) e aparece no manifesto JSON de cada figura, junto com hiperparâmetros, transformações e a definição de cada barra de incerteza. Onde aparece dispersão, ela é desvio-padrão entre repetições — **não** intervalo de confiança, e está dito assim no código. Pequenas variações numéricas entre versões de biblioteca são esperadas; as conclusões qualitativas não dependem delas.

## Decisões de figura

Paleta Okabe-Ito, segura para as formas mais comuns de daltonismo. A cor nunca é o único canal: marcadores, tracejados, hachuras e rótulos diretos são redundantes. Barras partem do zero. Eixos duplos aparecem uma única vez (demo 05) e com as duas escalas rotuladas. Projeções 2D genéricas de espaço químico foram **deliberadamente evitadas** no demo 01: t-SNE, UMAP e PCA comprimem dezenas de famílias em dois eixos e rotineiramente escondem justamente a novidade, que colapsa no centro. Os eixos usados são quantidades auditáveis.

## O que este repositório **não** demonstra

Enumerar isto é parte do trabalho, não uma ressalva de rodapé.

- **Nada sobre química amazônica real.** Nenhum número aqui é estimativa de quantos metabólitos da região estão descritos, nem de taxa de anotação em metabolômica não direcionada, nem de fração de fármacos derivados de produtos naturais. Esses números existem na literatura e **devem entrar no slide com fonte primária verificada, DOI conferido e data de acesso.**
- **Nenhuma validação experimental.** Detecção de BGC não é produção de metabólito; predição de permeabilidade não é atividade central; hit em triagem não é candidato a fármaco. A bancada continua sendo quem converte E5 em E3.
- **Nenhum conteúdo jurídico.** As regras do demo 08 não substituem a leitura do texto vigente da Convenção sobre Diversidade Biológica, do Protocolo de Nagoia, da Lei nº 13.123/2015 e sua regulamentação, e das normas do SisGen. O estado das negociações sobre informação de sequência digital (DSI) no âmbito da CDB muda, e **precisa ser verificado antes de qualquer afirmação pública.**
- **Nenhuma molécula gerada.** Modelos generativos e docagem estão fora do escopo deliberadamente: são a parte do campo em que resultados E5 mais frequentemente circulam como se fossem E3.

## Como citar

Veja `CITATION.cff`. Ao reutilizar qualquer figura, mantenha o carimbo de dados sintéticos.

## Licença

MIT — veja `LICENSE`. Uso livre, inclusive em aula e em material derivado, mantida a atribuição.

---

**Prof. Bruno Duarte Gomes**
Faculdade de Biotecnologia, ICB, UFPA
Laboratório de Neurofisiologia Eduardo Oswaldo Cruz (LNEOC) — ICB — UFPA
Laboratório de Simulação e Biologia Computacional (SimBic) — CCAD-IA, UFPA
