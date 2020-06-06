import src

content_path_list, style_path_list = src.st.data.get_data()

content_path, style_path, result_path, img_start = src.st.data.get_next_files(content_path_list, style_path_list)

if result_path is not None:
    # image_couple = src.images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)

    extractor = src.st.StyleContentModel(
        style_layers=src.st.var.p.style_layers,
        content_layers=src.st.var.p.content_layers
    )

    optimizers = src.st.Optimizers()

    result_path.mkdir()
    src.st.train.style_transfert(
        content_path=content_path,
        style_path=style_path,
        extractor=extractor,
        optimizers=optimizers,
        image_start=img_start
    )
else:
    print('No result_path left...')



