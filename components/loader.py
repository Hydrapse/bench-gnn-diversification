from torch_geometric.loader import NeighborLoader


def get_loader(config, data, n_mask):
    name = config.name.lower()
    assert name in ['sage']
    if name == 'sage':
        return NeighborLoader(
            data,
            input_nodes=n_mask,
            num_neighbors=config.fanouts,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )
    else:
        raise NotImplementedError
