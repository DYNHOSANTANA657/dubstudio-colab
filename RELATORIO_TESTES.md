# DubStudio Colab v3 — registro de validação

Data: 9 de outubro de 2026. Este relatório separa testes locais de execução real dos modelos.

## Evidências executadas

- A regressão local inicial de 85 testes passou, incluindo regras compartilhadas, pacotes/importação v2 e servidor HTTP. Os testes legados não são incluídos nesta distribuição limpa.
- A suíte do runner Colab de 32 testes passou, incluindo lip-sync opcional, integridade dos checkpoints, tradução, escolha facial, mídia sintética e compilação das células do notebook.
- Após incorporar a preferência de cache no Drive, 18 testes do bridge v2 passaram: preferência padrão ligada, ligada/desligada explícita, preservação no pacote revisado, consentimento, checksums, ZIP inseguro, vozes, lip-sync e mídia real sintética.
- Após incorporar essa preferência ao servidor, 16 testes HTTP passaram: três operações, configuração efetiva, revisão/retomada, resultados parciais, downloads, ranges, origens/Host, caminhos, restauração, arquivamento, booleans e coordenadas faciais.

Esses números descrevem rodadas de teste distintas e sobrepostas; não devem ser somados como se fossem casos exclusivos.

## Verificações do pacote portátil

`build_portable.py` usa lista explícita de arquivos públicos, sem inclusão recursiva. Depois de construir o ZIP, compara nomes, bytes de origem e SHA-256, rejeita links e verifica ausência de runtime, mídia privada, configuração privada, executáveis e arquivos RunPod/Docker. Produz um notebook separado idêntico à origem e checksums externos. Arquivos existentes no destino não são sobrescritos.

O ZIP contém `SHA256SUMS.txt` e `PACKAGING_REPORT.json` com o inventário público. A inclusão de código e testes não equivale à inclusão de vídeos ou resultados reais de usuários.

## Pendências importantes

- A instalação limpa e a inferência real no runtime gratuito do Colab com GPU ainda precisam ser validadas.
- Não há certificação de tradução em polonês, semelhança do timbre, qualidade visual ou sincronização labial apenas pelos testes sintéticos.
- A duração e integridade das faixas são verificadas automaticamente, mas cada resultado deve ser reproduzido e conferido pelo usuário.
- O Colab gratuito pode ficar sem GPU ou interromper a sessão. Nenhum prazo fixo de renderização é garantido.
- Lip-sync é opcional na dublagem e obrigatório na operação de sincronizar vídeo com áudio enviado. Quando desativado, o relatório não declara lip-sync confirmado.

## Reexecutar os testes incluídos

Requisitos locais: Python 3.11/3.12, FFmpeg e FFprobe no PATH.

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s colab -p test_runner_v2.py -v
```

Para montar outra distribuição limpa:

```powershell
python build_portable.py --output-dir "$env:USERPROFILE\Downloads"
```

O script não altera projetos, não inicia GPU, não envia mídia e não publica em contas externas.
