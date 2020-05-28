import src

content_path_list, style_path_list = src.data.get_data()

content_path, style_path, result_path = src.data.get_next_files(content_path_list, style_path_list)

if result_path is not None:
    # image_couple = src.images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)

    extractor = src.StyleContentModel(
        style_layers=src.p.style_layers,
        content_layers=src.p.content_layers
    )

    optimizers = src.Optimizers()

    result_path.mkdir()
    src.train.style_transfert(content_path, style_path, extractor, optimizers)
else:
    print('No result_path left...')



