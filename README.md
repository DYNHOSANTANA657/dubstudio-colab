# DubStudio Colab

Interface local em português; IA executada no notebook Colab manual, com timbre original autorizado e lip-sync opcional na dublagem.

Inicie por `start_windows.bat`. Requisitos: Python 3.11/3.12, FFmpeg e FFprobe. O guia completo, limites e estado de validação estão em [README_COLAB.md](README_COLAB.md).

**Validação real parcial em uma T4 do Colab:** dublagem português → polonês de um trecho de 20 segundos sem lip-sync e Texto para voz foram executados, importados e baixados pelo Studio. Lip-sync e os demais cenários de aceite continuam em validação. Consulte [RELATORIO_TESTES.md](RELATORIO_TESTES.md). Os testes locais de arquivos não certificam timbre ou sincronização perceptual.

Nenhum vídeo, referência vocal, resultado ou credencial deve ser publicado no GitHub. O Colab não garante GPU gratuita e proíbe deepfakes em runtimes gerenciados; não há túnel, keepalive ou contorno de cotas.
