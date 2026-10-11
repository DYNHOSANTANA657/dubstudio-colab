# DubStudio Colab

Interface local em português; IA executada no notebook Colab manual, com timbre original autorizado e lip-sync opcional na dublagem.

O padrão dos novos projetos é **áudio natural como mestre**: a voz traduzida não é acelerada, encurtada nem cortada para caber no vídeo original. O vídeo é cortado ou repete trechos da pessoa correspondente para acompanhar cada fala; cada idioma pode ter duração final diferente. Com lip-sync ligado, a boca é gerada depois dessa montagem. Resultados antigos continuam preservados com seu contrato original.

Inicie por `start_windows.bat`. Requisitos: Python 3.11/3.12, FFmpeg e FFprobe. O guia completo, limites e estado de validação estão em [README_COLAB.md](README_COLAB.md).

**Validação real parcial:** as execuções anteriores exercitaram dublagem, Texto para voz e lip-sync com áudio enviado, mas não aprovam retroativamente o novo modo de montagem nem a composição perioral mais recente. O vídeo de aproximadamente 80 segundos teve integridade técnica conferida, mas apresentou defeito visual e aceleração vocal no modo antigo. O novo modo exige reteste audiovisual completo. Consulte [RELATORIO_TESTES.md](RELATORIO_TESTES.md): suporte do motor e testes técnicos não certificam todos os idiomas, timbre idêntico ou sincronismo perceptual perfeito.

Nenhum vídeo, referência vocal, resultado ou credencial deve ser publicado no GitHub. O Colab não garante GPU gratuita e proíbe deepfakes em runtimes gerenciados; não há túnel, keepalive ou contorno de cotas.
