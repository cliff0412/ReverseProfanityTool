from src.tools import RevProfanity
import pyopencl as cl


if __name__ == "__main__":

    # List GPU memory info
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            print(f"Device name: {device.name}")
            print(f"Device type: {cl.device_type.to_string(device.type)}")
            print(f"Device memory: {device.global_mem_size / 1024 ** 2} MB")

    prof = RevProfanity('ethereum')

    pct_range = int(
        round((int(device.global_mem_size * 0.8) / 1024 ** 2) / (32 * 1024), 2) * 100)  # Calculate the % of output file
    iter_properties = [(i, min(i + pct_range, 100), 'output.bin') for i in range(0, 100, pct_range)]


    for (st, en, str_nam) in iter_properties:

        addresses = []
        contracts = []
        print('Initializing pubKeys...')
        prof.load_sorted_keys(st, en, str_nam)

        eoas = prof.get_deployers(contracts)
        addresses.extend(eoas)
        addresses, txs = prof.get_tx_etherscan(addresses)
        addresses = prof.get_pubkeys(addresses, txs)
        print('Number of addresses: ', addresses.shape[0])

        prof.build_kernel()
        prof.run_kernel(addresses.address.to_list(), addresses.public_key.to_list())
