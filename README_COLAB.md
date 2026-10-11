# DubStudio Colab 3

Studio em português com interface local e computação em notebook Colab manual. O computador prepara arquivos, permite revisar a tradução e valida resultados; não executa os modelos de IA.

## Estado desta entrega

Os testes reais anteriores exercitaram dublagem de um trecho de 20 segundos para polonês, inglês e espanhol, Texto para voz em polonês/espanhol e lip-sync com áudio enviado, incluindo importação, reprodução e downloads no Studio. São evidências históricas: não aprovam automaticamente o novo modo de áudio natural nem a máscara perioral mais recente. O vídeo de aproximadamente 80 segundos passou em integridade técnica, mas a inspeção encontrou alteração em olhos/testa e falas aceleradas no modo antigo. Consulte [RELATORIO_TESTES.md](RELATORIO_TESTES.md). Suporte declarado pelo modelo não significa idioma aprovado em teste. Não há garantia de timbre idêntico, prazo de processamento ou GPU gratuita disponível.

## Áudio natural, vídeo adaptado

Nos novos projetos, o áudio final define a duração. As falas completas são geradas com o timbre autorizado de cada pessoa, sem `atempo`, aceleração, desaceleração, corte de palavras ou resumo para caber nos tempos originais. A taxa de amostragem pode ser convertida para exportação; isso não muda a velocidade da fala.

O sistema mantém a ordem dos turnos: pessoa A → B → A continua nessa ordem. Se uma fala traduzida for maior, repete trechos de vídeo da mesma pessoa, para a frente e em velocidade normal; se for menor, corta o excesso visual. Quando há sobreposição de vozes no original, as falas confirmadas são colocadas em sequência, uma voz por vez. Uma voz ou rosto ambíguo exige correção em vez de associação inventada.

Com lip-sync ligado, a montagem visual acontece primeiro; a boca é então gerada para o áudio completo de cada turno. Um vídeo já sincronizado não é repetido para cobrir palavras diferentes. Com lip-sync desligado, a mesma montagem é feita, mas os movimentos de boca não são alterados. Identificar a pessoa correspondente continua necessário para a montagem; desligar lip-sync evita MuseTalk, não autoriza mostrar uma pessoa diferente daquela que está falando. Música e ambiente acompanham a nova duração sem repetir a voz original.

Os tempos da transcrição original e da revisão aprovada permanecem intactos. Cada idioma ganha sua própria linha de saída e legendas nos novos tempos. O vídeo final deve terminar junto com seu áudio, dentro da precisão do quadro e do codec — não é obrigado a ter a duração original. Resultados antigos continuam com a regra e as evidências que usavam quando foram criados.

## Abrir o Studio no Windows

Requisitos: Python 3.11 ou 3.12, FFmpeg/FFprobe no PATH e navegador moderno.

Execute `start_windows.bat`. O Studio abre em http://127.0.0.1:8765 e não é exposto na rede. O servidor roda oculto; logs ficam em `runtime/studio-*.log`. Reabrir o atalho reutiliza a interface já iniciada.

## Usar

1. Em **Dublar vídeo**, escolha o original e um ou mais idiomas com suporte a timbre original. Confirme a autorização de uso da mídia e da voz.
2. Escolha revisão antes da voz ou execução automática. Música/ambiente são preservados quando selecionados. Na dublagem, lip-sync inicia ligado e pode ser desligado. O vídeo acompanha as falas naturais nos dois casos. Desligado, não há alteração facial; ligado, não há entrega silenciosa sem lip-sync: ambiguidades interrompem a etapa. O relatório informa se foi solicitado e aplicado.
3. **Preparar projeto** gera `pacote-colab.zip`. **Abrir no Colab** baixa o pacote e abre o notebook publicado configurado. Se houver bloqueio de popup, use os dois botões separados. O notebook também pode ser baixado e aberto manualmente.
4. No Colab, escolha GPU, conecte e execute as células. Envie o pacote pelo seletor. A verificação da GPU ocorre antes dos downloads pesados.
5. Acompanhe as etapas reais no notebook. A interface local não inventa porcentagens nem se conecta ao runtime por túnel.
6. Para revisar traduções, importe o ZIP intermediário no Studio, edite apenas os textos traduzidos e exporte o pacote aprovado. Reenvie-o ao notebook para retomar as etapas afetadas.
7. Importe o ZIP final em **Minhas criações**. Cada idioma é conferido separadamente. Versões incompletas ou trocadas não recebem download aprovado.
8. Reproduza a versão aprovada para conferir tradução, timbre e sincronização. O relatório técnico não substitui esta conferência perceptiva.

**Lip-sync com áudio** usa vídeo e áudio enviados, sem traduzir ou trocar a voz. O áudio enviado mantém sua duração natural, e o vídeo é adaptado a ele. Seleção manual de rosto precisa de coordenadas reais; um rosto ambíguo ou várias vozes sem associação confiável interrompem a etapa.

Se uma análise facial interromper o projeto, use **Selecionar rosto para retomada** na criação correspondente. Escolha o tempo da cena e delimite o rosto correto na prévia; salve e use **Preparar pacote corrigido**. A seleção não inicia processamento: baixe o novo pacote e envie-o ao Colab. O pacote antigo fica preservado, mas não deve ser usado para essa retomada. Uma cena curta exige confirmação explícita e evidência consistente; o sistema não ignora automaticamente a cena nem entrega uma versão sem o lip-sync solicitado.

**Texto para voz** exige texto, idioma compatível e referência vocal autorizada, com WAV e MP3. Não utiliza voz genérica como fallback.

## Cache e retomada

Os modelos ficam em cache no Drive somente se o usuário autorizar a montagem. Os ambientes Python são separados por motor e preparados uma vez por sessão em `/content`. Pesos/checkpoints podem persistir no Drive; ambientes incompatíveis não são reaproveitados entre sessões. Cada etapa registra identidade, opções, hashes e saídas. Alterar a tradução invalida a geração posterior afetada; alterar o código do pipeline também invalida os checkpoints incompatíveis. O modo de montagem integra a identidade das etapas: áudio ajustado e lip-sync antigos não passam a valer como áudio natural. Áudios RAW completos só podem ser reutilizados quando texto, idioma, locutor, referência e motor continuarem vinculados. Uma falha preserva etapas anteriores íntegras, mas não transforma arquivo antigo em resultado atual. Não há keepalive ou tentativa de contornar cotas.

Para liberar o disco temporário, não apague a pasta inteira de execução nem os modelos principais. Cópias redundantes só devem ser removidas depois de confirmar seus alvos e a integridade das cópias preservadas. Limpar `/content` não é limpar o Drive; checkpoints e resultados devem ser preservados antes de qualquer limpeza. Instalação inicial, inferência e recomposição são tempos diferentes, registrados separadamente.

## Limites e segurança

- Primeira entrega: prioridade a vídeos de até 2 minutos; limite de mídia de 20 minutos e upload de 8 GB. Uploads grandes também dependem da memória e cotas do Colab.
- Exportação H.264/AAC, sem ampliar a imagem e no máximo 1080p. No modo novo, as faixas são medidas separadamente contra o áudio mestre e a linha de saída. A diferença de duração em relação ao vídeo original é esperada; o final de áudio/vídeo segue a tolerância explícita de quadros/codec. O contrato legado conserva sua própria verificação.
- O modo novo não oferece suporte a taxa variável de quadros (VFR). Quando a divergência é detectada pelos metadados, a montagem é interrompida e pede normalização explícita; o original e checkpoints íntegros são preservados. Não há conversão silenciosa nem alegação de detector universal de irregularidades de PTS.
- ZIPs são verificados contra travessia de diretório, links simbólicos, nomes duplicados, tamanho excessivo, identificadores divergentes e hashes incorretos.
- Criação e importação são locais. Código público não contém mídia, configurações privadas, referências de voz, resultados ou credenciais.
- Arquivar uma criação preserva sua pasta em `runtime/jobs`. A pasta antiga e os volumes não são apagados na migração.
- **Excluir projeto** pede confirmação e move somente a cópia local para `runtime/trash`. Em **Minhas criações → Lixeira → Restaurar projeto**, os arquivos podem voltar à biblioteca. Downloads e originais externos ficam intactos; a exclusão não cancela uma execução no Colab e não esvazia a lixeira automaticamente.
- RunPod, Docker e demonstração não são iniciados por esta interface. O servidor antigo foi preservado como `server_runpod_legacy.py` apenas para histórico, não como entrada do app.

## Regras do Colab e dos modelos

O Colab gratuito não garante GPU nem duração de sessão. A FAQ proíbe criação de deepfakes em runtimes gerenciados. Autorização vocal não equivale a autorização do Google para uma operação facial. Verifique as regras do serviço; um bloqueio deve ser registrado como falha, preservar checkpoints e nunca ser contornado ou apresentado como conclusão.

Fontes oficiais: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Chatterbox](https://github.com/resemble-ai/chatterbox), [MuseTalk](https://github.com/TMElyralab/MuseTalk), [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt). Respeite também as licenças dos pesos e datasets usados pelos motores.

## Código e testes

Entradas: `server.py` → `studio_server.py`; bridge `studio_bridge.py`; notebook `DubStudio_Colab.ipynb`; runner `colab/dubstudio_runner_v2.py`; montagem natural `colab/dubstudio_audio_master.py`; preparo `colab/dubstudio_bootstrap.py`.

Manifesto v2: operação, idiomas, entradas/checksums, autorização, opções, modo temporal e versão do pipeline. Resultado `result.json`: estado e arquivos/checksums por idioma, incluindo áudio mestre e mapa de montagem no novo modo. Revisão `review.json` → `review-approved.json`, sem alterar falantes ou tempos originais; os tempos finais ficam em evidência separada.

Execute os testes locais com `python -m unittest discover -s tests -v`. Os testes com mídia sintética validam as regras de integridade; não são uma validação real dos modelos na GPU.

Rodada local desta integração: 623 métodos incluídos no pacote, 619 aprovados e quatro ignorados por privilégio de symlink no Windows, sem falhas ou erros. FFmpeg e contratos foram exercitados com mídias sintéticas; etapas de IA simuladas não aprovam tradução, timbre ou lip-sync perceptual. Veja o relatório para escopo, hashes e pendências.
