# DubStudio Colab 3

Studio em português com interface local e computação em notebook Colab manual. O computador prepara arquivos, permite revisar a tradução e valida resultados; não executa os modelos de IA.

## Estado desta entrega

Os contratos e as proteções locais estão em testes. Em uma T4 do Colab, um trecho de 20 segundos foi dublado de português para polonês sem lip-sync, importado, reproduzido e baixado pelo Studio; Texto para voz também foi executado e importado. Lip-sync e os demais cenários de aceite continuam em validação: consulte [RELATORIO_TESTES.md](RELATORIO_TESTES.md). Suporte declarado pelo modelo não significa idioma já aprovado em teste. Não há garantia de timbre idêntico, prazo de processamento ou GPU gratuita disponível.

## Abrir o Studio no Windows

Requisitos: Python 3.11 ou 3.12, FFmpeg/FFprobe no PATH e navegador moderno.

Execute `start_windows.bat`. O Studio abre em http://127.0.0.1:8765 e não é exposto na rede. O servidor roda oculto; logs ficam em `runtime/studio-*.log`. Reabrir o atalho reutiliza a interface já iniciada.

## Usar

1. Em **Dublar vídeo**, escolha o original e um ou mais idiomas com suporte a timbre original. Confirme a autorização de uso da mídia e da voz.
2. Escolha revisão antes da voz ou execução automática. Música/ambiente são preservados quando selecionados. Na dublagem, lip-sync inicia ligado e pode ser desligado. Desligado, o vídeo mantém os movimentos originais e recebe somente o áudio dublado; etapas e modelos faciais são pulados. Ligado, não há entrega silenciosa sem lip-sync: ambiguidades interrompem a etapa. O relatório informa se foi solicitado e aplicado.
3. **Preparar projeto** gera `pacote-colab.zip`. **Abrir no Colab** baixa o pacote e abre o notebook publicado configurado. Se houver bloqueio de popup, use os dois botões separados. O notebook também pode ser baixado e aberto manualmente.
4. No Colab, escolha GPU, conecte e execute as células. Envie o pacote pelo seletor. A verificação da GPU ocorre antes dos downloads pesados.
5. Acompanhe as etapas reais no notebook. A interface local não inventa porcentagens nem se conecta ao runtime por túnel.
6. Para revisar traduções, importe o ZIP intermediário no Studio, edite apenas os textos traduzidos e exporte o pacote aprovado. Reenvie-o ao notebook para retomar as etapas afetadas.
7. Importe o ZIP final em **Minhas criações**. Cada idioma é conferido separadamente. Versões incompletas ou trocadas não recebem download aprovado.
8. Reproduza a versão aprovada para conferir tradução, timbre e sincronização. O relatório técnico não substitui esta conferência perceptiva.

**Lip-sync com áudio** usa vídeo e áudio enviados, sem traduzir ou trocar a voz. Seleção manual de rosto precisa de coordenadas reais; um rosto ambíguo interrompe a etapa.

**Texto para voz** exige texto, idioma compatível e referência vocal autorizada, com WAV e MP3. Não utiliza voz genérica como fallback.

## Cache e retomada

Os modelos ficam em cache no Drive somente se o usuário autorizar a montagem. Os ambientes Python são separados por motor e preparados uma vez por sessão em `/content`. Pesos/checkpoints podem persistir no Drive; ambientes incompatíveis não são reaproveitados entre sessões. Cada etapa registra identidade, opções, hashes e saídas. Alterar a tradução invalida somente a geração posterior afetada. Não há keepalive ou tentativa de contornar cotas.

## Limites e segurança

- Primeira entrega: prioridade a vídeos de até 2 minutos; limite de mídia de 20 minutos e upload de 8 GB. Uploads grandes também dependem da memória e cotas do Colab.
- Exportação H.264/AAC, sem ampliar a imagem e no máximo 1080p; duração das faixas visual e sonora medida separadamente, tolerância máxima de 1 segundo.
- ZIPs são verificados contra travessia de diretório, links simbólicos, nomes duplicados, tamanho excessivo, identificadores divergentes e hashes incorretos.
- Criação e importação são locais. Código público não contém mídia, configurações privadas, referências de voz, resultados ou credenciais.
- Arquivar uma criação preserva sua pasta em `runtime/jobs`. A pasta antiga e os volumes não são apagados na migração.
- RunPod, Docker e demonstração não são iniciados por esta interface. O servidor antigo foi preservado como `server_runpod_legacy.py` apenas para histórico, não como entrada do app.

## Regras do Colab e dos modelos

O Colab gratuito não garante GPU nem duração de sessão. A FAQ proíbe criação de deepfakes em runtimes gerenciados. Autorização vocal não equivale a autorização do Google para uma operação facial. Verifique as regras do serviço; um bloqueio deve ser registrado como falha, preservar checkpoints e nunca ser contornado ou apresentado como conclusão.

Fontes oficiais: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Chatterbox](https://github.com/resemble-ai/chatterbox), [MuseTalk](https://github.com/TMElyralab/MuseTalk), [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt). Respeite também as licenças dos pesos e datasets usados pelos motores.

## Código e testes

Entradas: `server.py` → `studio_server.py`; bridge novo `studio_bridge.py`; notebook `DubStudio_Colab.ipynb`; runner `colab/dubstudio_runner_v2.py`; preparo `colab/dubstudio_bootstrap.py`.

Manifesto v2: operação, idiomas, entradas/checksums, autorização, opções e versão do pipeline. Resultado `result.json`: estado e arquivos/checksums por idioma. Revisão `review.json` → `review-approved.json`, sem alterar falantes ou tempos originais.

Execute os testes locais com `python -m unittest discover -s tests -v`. Os testes com mídia sintética validam as regras de integridade; não são uma validação real dos modelos na GPU.
