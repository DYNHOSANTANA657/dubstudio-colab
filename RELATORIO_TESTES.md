# DubStudio Colab v3 — registro de validação

Data: 10 de outubro de 2026. Este relatório separa testes locais de execução real dos modelos.

## Evidências executadas

- Rodada consolidada Windows mais recente: 124 testes distintos, 120 aprovados, 4 ignorados e zero falhas. Colab: 79 testes em 8,610 segundos; Studio: 45 testes aprovados em 31,992 segundos. Os quatro ignorados exigem links simbólicos reais, indisponíveis no Windows sem privilégio. No reteste Linux da sessão Colab, os 19 casos de cache passaram em 0,052 segundo, incluindo os quatro ignorados no Windows.
- Novos testes com WAV e FFmpeg reais cobrem tempos de TTS, reaproveitamento do áudio original sem corte para ajustes de duração e recusa de importação de uma tradução aprovada antiga. Esses testes locais não equivalem a síntese ou lip-sync GPU real.
- A regressão local inicial de 85 testes passou, incluindo regras compartilhadas, pacotes/importação v2 e servidor HTTP. Os testes legados não são incluídos nesta distribuição limpa.
- A suíte do runner Colab de 32 testes passou, incluindo lip-sync opcional, integridade dos checkpoints, tradução, escolha facial, mídia sintética e compilação das células do notebook.
- Após incorporar a preferência de cache no Drive, 18 testes do bridge v2 passaram: preferência padrão ligada, ligada/desligada explícita, preservação no pacote revisado, consentimento, checksums, ZIP inseguro, vozes, lip-sync e mídia real sintética.
- Após incorporar essa preferência ao servidor, 16 testes HTTP passaram: três operações, configuração efetiva, revisão/retomada, resultados parciais, downloads, ranges, origens/Host, caminhos, restauração, arquivamento, booleans e coordenadas faciais.

Esses números descrevem rodadas de teste distintas e sobrepostas; não devem ser somados como se fossem casos exclusivos.

## Reteste e diagnóstico no Colab real

- A instalação inicial foi concluída em uma sessão com Tesla T4 (15 GB), sem substituir o runtime do Colab ou automatizar autenticação.
- Separação Demucs: 9,0 segundos. Faster-Whisper: 17,6 segundos para transcrever a entrada de 20 segundos. Identificação de um locutor e extração da referência vocal concluídas.
- MADLAD apresentou geração repetitiva sem EOS. O diagnóstico mostrou Transformers 5.2.0, tokenizer/EOS corretos e logits finitos. A correção upstream [PR #48238](https://github.com/huggingface/transformers/pull/48238) reordena o carregamento T5; o reteste comprovou embeddings de encoder/decoder iguais, projeção de saída distinta e geração em polonês terminada pelo EOS 2.
- Esse primeiro reteste não aprovou fidelidade semântica: "reais" virou "dólares". A tradução completa dos três trechos foi gerada na T4 em 91,1 segundos e revisada no Studio, preservando a moeda original. Importação, edição e exportação da revisão foram exercitadas pela interface. Geração em polonês não equivale a tradução correta.
- A retomada revelou um erro real: uma nova transcrição deslocou o início do último trecho de 18,74 para 18,76 segundos e invalidou a revisão aprovada. A correção passou no reteste real: o pacote revisado contém a transcrição original congelada, vinculada à aprovação por SHA-256, e a retomada não repete ASR/diarização nem amplia a tolerância.
- Um teste isolado do Chatterbox Multilingual concluiu os três trechos revisados na GPU. As falas brutas duraram 7,8, 12,32 e 1,8 segundos, ajustadas para suas janelas sem cortar o final. FFprobe confirmou WAV estéreo de 48 kHz com 20,000 segundos. Uma segunda transcrição na GPU confirmou os três trechos em polonês, incluindo o final detectado em 19,7 segundos. Essa auditoria de conteúdo não certifica semelhança perceptual do timbre.
- O fluxo completo do trecho de 20 segundos concluiu a retomada aprovada em 83,653 segundos: separação 10,973; referências 0,397; síntese 65,434; mixagem 1,357; exportação 4,412. Esses tempos não incluem a instalação inicial nem a tradução/revisão anteriores. A síntese registrou pico de uso GPU de 97% e 4.029 MiB de memória.
- O resultado real sem lip-sync foi importado no Studio. Os seis downloads (vídeo, WAV, MP3, duas legendas e relatório) funcionaram; os bytes do vídeo baixado coincidiram com os importados. A reprodução chegou ao fim sem erro. FFprobe: vídeo 20,000 segundos, áudio 19,916 segundos, diferença 0,084 segundo, imagem 360×640 preservada. O relatório declara corretamente que lip-sync não foi aplicado.
- Uma auditoria independente do mix final na T4 identificou os trechos em polonês, a moeda brasileira preservada e a última fala em 18,32–19,88 segundos. A entrada de teste foi recortada em 20 segundos e termina no meio de uma frase; o final parcial corresponde a esse recorte. Isso não valida ainda o vídeo original de aproximadamente 80 segundos nem substitui a conferência humana do timbre e da pronúncia.
- Texto para voz foi exercitado pela interface: texto curto em polonês, referência vocal autorizada, exportação do pacote, síntese na T4 (45,4 segundos), importação no Studio e downloads WAV/MP3/relatório. O áudio tem 3,56 segundos, carrega sem erro e a auditoria independente reconheceu a narração em polonês. Há diferença de reconhecimento no nome "DubStudio"; este teste não certifica pronúncia ou timbre por comparação auditiva humana.
- O cache sem Drive foi corrigido para usar o próprio runtime, sem duplicar pesos/checkpoints. Cache persistente preserva blobs únicos e referências de snapshots, inclusive quando o Drive não suporta links simbólicos.
- Servidor local: 35 testes bridge/HTTP passaram, incluindo bloqueio de um segundo listener na mesma porta do Windows. Uma única instância do Studio ficou ativa.
- Cache: 19 métodos, 15 aprovados e 4 dependentes de symlink real pulados por restrição de privilégio no Windows; os casos de Drive sem symlink foram simulados e passaram. Runner: 32 aprovados. Carregamento MADLAD: 4 regressões unitárias aprovadas. Esses testes não certificam qualidade vocal ou visual.

## Verificações do pacote portátil

`build_portable.py` usa lista explícita de arquivos públicos, sem inclusão recursiva. Depois de construir o ZIP, compara nomes, bytes de origem e SHA-256, rejeita links e verifica ausência de runtime, mídia privada, configuração privada, executáveis e arquivos RunPod/Docker. Produz um notebook separado idêntico à origem e checksums externos. Arquivos existentes no destino não são sobrescritos.

O ZIP contém `SHA256SUMS.txt` e `PACKAGING_REPORT.json` com o inventário público. A inclusão de código e testes não equivale à inclusão de vídeos ou resultados reais de usuários.

## Pendências importantes

- Instalação, separação, transcrição, tradução, síntese, retomada aprovada, auditoria do áudio, exportação, importação, reprodução e downloads foram exercitados na T4 e no Studio para o trecho de 20 segundos sem lip-sync. Texto para voz também gerou um resultado real importado e auditado. Lip-sync real, vídeo original completo e múltiplos idiomas/locutores continuam em validação.
- Não há certificação de tradução em polonês, semelhança do timbre, qualidade visual ou sincronização labial apenas pelos testes sintéticos.
- A duração e integridade das faixas são verificadas automaticamente, mas cada resultado deve ser reproduzido e conferido pelo usuário.
- O Colab gratuito pode ficar sem GPU ou interromper a sessão. Nenhum prazo fixo de renderização é garantido.
- Lip-sync é opcional na dublagem e obrigatório na operação de sincronizar vídeo com áudio enviado. Quando desativado, o relatório não declara lip-sync confirmado.

## Reexecutar os testes incluídos

Requisitos locais: Python 3.11/3.12, FFmpeg e FFprobe no PATH.

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s colab -p "test_*.py" -v
```

Para montar outra distribuição limpa:

```powershell
python build_portable.py --output-dir "$env:USERPROFILE\Downloads"
```

O script não altera projetos, não inicia GPU, não envia mídia e não publica em contas externas.
