import src

content_path_list, style_path_list = src.st.data.get_data()

content_path, style_path, result_path, img_start = src.st.data.get_next_files(content_path_list, style_path_list)

if result_path is not None:
    # image_couple = src.images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)

    extractor = src.st.StyleContentModel(
        style_layers=src.st.var.style_layers,
        content_layers=src.st.var.content_layers
    )

    optimizers = src.st.Optimizers(
        shape=(1,),
        lr=src.st.var.lr
    )

    result_path.mkdir()
    src.st.train.style_transfert(
        content_path=content_path,
        style_path=style_path,
        extractor=extractor,
        optimizers=optimizers,
        image_start=img_start,
        epochs=src.st.var.epochs,
        steps_per_epoch=src.st.var.epochs
    )
else:
    print('No result_path left...')



