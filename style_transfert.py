import src

content_path_list, style_path_list = src.st.data.get_data()

file_combination = src.st.data.get_next_files(content_path_list, style_path_list)

if file_combination is not None:
    # image_couple = src.images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)

    extractor = src.st.StyleContentModel(
        style_layers=src.st.var.style_layers,
        content_layers=src.st.var.content_layers,
        content_gram_layers=src.st.var.content_gram_layers
    )

    optimizers = src.st.Optimizers(
        shape=(1,),
        lr=src.st.var.lr
    )

    file_combination.results_folder.mkdir()
    src.st.train.style_transfert(
        file_combination=file_combination,
        extractor=extractor,
        optimizers=optimizers,
        epochs=src.st.var.epochs,
        steps_per_epoch=src.st.var.steps_per_epoch
    )
else:
    print('No result_path left...')



